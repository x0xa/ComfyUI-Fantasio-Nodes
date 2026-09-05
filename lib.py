import io
import os
import ssl
import time
import uuid
import errno
import shutil
import socket
import tarfile
import zipfile
import tempfile
import http.client
import urllib.parse

import boto3
import numpy as np
from PIL import Image
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

DOWNLOAD_CHUNK_SIZE = 1024 * 512
UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY_SECONDS = 3
DOWNLOAD_USER_AGENT = "fantasio-comfy-node/1.0"
CONNECT_TIMEOUT_SECONDS = 5
CONNECT_ATTEMPTS = 2
MAX_REDIRECT_HOPS = 5
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
EGRESS_PROBE_HOST = "1.1.1.1"
EGRESS_PROBE_PORT = 443
EGRESS_PROBE_TIMEOUT_SECONDS = 1

EGRESS_ERRNOS = frozenset({
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ENETDOWN,
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ECONNABORTED,
    errno.EPIPE,
})

RESOLVER_FAILURE_ERRNOS = frozenset(
    code for code in (
        getattr(socket, 'EAI_AGAIN', None),
        getattr(socket, 'EAI_FAIL', None),
        getattr(socket, 'EAI_NODATA', None),
        getattr(socket, 'EAI_SYSTEM', None),
    ) if code is not None
)


class DownloadDeadlineExceeded(Exception):
    pass


class InstanceEgressUnavailable(Exception):
    pass


def describe_exception(error):
    text = str(error).strip()
    name = type(error).__name__
    lowered = text.lower()

    if name == "OutOfMemoryError" or "out of memory" in lowered or "cuda error" in lowered:
        return f"CUDA out of memory / GPU error: {text}"

    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code")
        return f"S3 client error ({code}): {text}" if code else f"S3 client error: {text}"

    if isinstance(error, BotoCoreError):
        return f"S3 connection error ({name}): {text}"

    return f"{name}: {text}"


def create_s3_client(s3_endpoint, s3_access_key, s3_secret_key):
    return boto3.client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version='s3v4'),
        region_name='auto',
    )


def normalize_s3_public_url(base_url, key):
    return f"{base_url.rstrip('/')}/{key.lstrip('/')}"


def content_type_for(filename):
    return 'image/webp' if filename.lower().endswith('.webp') else 'application/octet-stream'


def _is_resolver_failure(error):
    return isinstance(error, socket.gaierror) and error.errno in RESOLVER_FAILURE_ERRNOS


def _is_transport_error(error):
    if isinstance(error, ssl.SSLCertVerificationError):
        return False

    if isinstance(error, socket.gaierror):
        return _is_resolver_failure(error)

    if isinstance(error, (socket.timeout, TimeoutError, ssl.SSLError, http.client.HTTPException)):
        return True

    return isinstance(error, OSError) and error.errno in EGRESS_ERRNOS


def _egress_probe_reachable():
    try:
        with socket.create_connection((EGRESS_PROBE_HOST, EGRESS_PROBE_PORT), timeout=EGRESS_PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def _raise_transfer_failure(url, error):
    if isinstance(error, socket.gaierror) and not _is_resolver_failure(error):
        raise RuntimeError(f"Hostname does not resolve for {url}: {describe_exception(error)}") from error

    if not _is_transport_error(error):
        raise RuntimeError(f"Failed to download {url}: {describe_exception(error)}") from error

    if _is_resolver_failure(error):
        raise InstanceEgressUnavailable(
            f"Instance DNS resolution failed for {url}: {describe_exception(error)}"
        ) from error

    if _egress_probe_reachable():
        raise RuntimeError(
            f"Origin unreachable while instance egress is healthy for {url}: {describe_exception(error)}"
        ) from error

    raise InstanceEgressUnavailable(
        f"Instance egress unavailable for {url}, probe {EGRESS_PROBE_HOST}:{EGRESS_PROBE_PORT} also unreachable: "
        f"{describe_exception(error)}"
    ) from error


def _create_connection(parsed, connect_timeout):
    if parsed.scheme == 'https':
        return http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port,
            timeout=connect_timeout,
            context=ssl.create_default_context(),
        )

    if parsed.scheme == 'http':
        return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=connect_timeout)

    raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")


def _request_target(parsed):
    target = parsed.path or '/'

    if parsed.query:
        return f"{target}?{parsed.query}"

    return target


def _open_hop(url, hop_url, read_timeout):
    parsed = urllib.parse.urlsplit(hop_url)
    connect_timeout = min(CONNECT_TIMEOUT_SECONDS, read_timeout)
    last_error = None

    for attempt in range(CONNECT_ATTEMPTS):
        connection = _create_connection(parsed, connect_timeout)

        try:
            connection.connect()
            connection.sock.settimeout(read_timeout)
            connection.request('GET', _request_target(parsed), headers={'User-Agent': DOWNLOAD_USER_AGENT})
            return connection, connection.getresponse(), parsed.hostname
        except Exception as error:
            connection.close()

            if not _is_transport_error(error) or attempt == CONNECT_ATTEMPTS - 1:
                _raise_transfer_failure(url, error)

            last_error = error

    _raise_transfer_failure(url, last_error)


def _open_response(url, read_timeout):
    hop_url = url

    for _ in range(MAX_REDIRECT_HOPS + 1):
        connection, response, hostname = _open_hop(url, hop_url, read_timeout)

        if response.status == 200:
            return connection, response

        status, reason = response.status, response.reason
        location = response.getheader('Location') if status in REDIRECT_STATUSES else None
        connection.close()

        if status in REDIRECT_STATUSES and not location:
            raise RuntimeError(f"Redirect without a target downloading {url}: {status} {reason} from {hostname}")

        if not location:
            raise RuntimeError(f"HTTP error downloading {url}: {status} {reason} from {hostname}")

        hop_url = urllib.parse.urljoin(hop_url, location)

    raise RuntimeError(f"Exceeded {MAX_REDIRECT_HOPS} redirects downloading {url}")


def _consume_response(response, url, write_chunk, progress_cb, deadline_seconds, started_at):
    total = response.getheader('Content-Length')
    total_bytes = int(total) if total else None
    downloaded = 0
    last_percent = 0

    while True:
        if deadline_seconds is not None and time.monotonic() - started_at > deadline_seconds:
            raise DownloadDeadlineExceeded(f"Download exceeded {deadline_seconds}s wall-clock limit for {url}")

        try:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
        except Exception as error:
            _raise_transfer_failure(url, error)

        if not chunk:
            break

        write_chunk(chunk)
        downloaded += len(chunk)

        if total_bytes and progress_cb:
            percent = int((downloaded / total_bytes) * 100)
            if percent > last_percent and percent < 100:
                progress_cb(percent)
                last_percent = percent

    if progress_cb:
        progress_cb(100)


def download_to_file(url, output_path, timeout_seconds=60, progress_cb=None, deadline_seconds=None):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    read_timeout = timeout_seconds
    if deadline_seconds is not None:
        read_timeout = min(timeout_seconds, deadline_seconds)

    started_at = time.monotonic()
    connection, response = _open_response(url, read_timeout)

    try:
        with open(output_path, 'wb') as output_file:
            _consume_response(response, url, output_file.write, progress_cb, deadline_seconds, started_at)
    except Exception:
        _remove_if_exists(output_path)
        raise
    finally:
        connection.close()


def download_bytes(url, timeout_seconds=60, progress_cb=None):
    connection, response = _open_response(url, timeout_seconds)
    buffer = io.BytesIO()

    try:
        _consume_response(response, url, buffer.write, progress_cb, None, time.monotonic())
    finally:
        connection.close()

    return buffer.getvalue()


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def _validate_archive_member(member_name, output_dir):
    if os.path.isabs(member_name):
        raise ValueError(f"Path traversal detected: absolute path '{member_name}'")
    if '..' in member_name.split(os.sep) or '..' in member_name.split('/'):
        raise ValueError(f"Path traversal detected: '..' in path '{member_name}'")
    resolved = os.path.normpath(os.path.join(output_dir, member_name))
    if not resolved.startswith(os.path.normpath(output_dir)):
        raise ValueError(f"Path traversal detected: '{member_name}' escapes output directory")


def _extract_archive(archive_path, extract_dir):
    lower_name = archive_path.lower()

    if lower_name.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                _validate_archive_member(member, extract_dir)
            zip_ref.extractall(extract_dir)
    elif lower_name.endswith('.tar.gz') or lower_name.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            for member in tar_ref.getnames():
                _validate_archive_member(member, extract_dir)
            tar_ref.extractall(extract_dir)
    elif lower_name.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            for member in tar_ref.getnames():
                _validate_archive_member(member, extract_dir)
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_and_extract_archive(url, output_dir, archive_name="", timeout_seconds=120, progress_cb=None):
    temp_dir = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="fantasio_archive_")

        if archive_name.strip():
            filename = archive_name.strip()
        else:
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path) or f"archive_{uuid.uuid4().hex}.zip"

        archive_path = os.path.join(temp_dir, filename)
        download_to_file(url, archive_path, timeout_seconds=timeout_seconds, progress_cb=progress_cb)

        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        _extract_archive(archive_path, extract_dir)

        os.makedirs(output_dir, exist_ok=True)
        for item in os.listdir(extract_dir):
            src = os.path.join(extract_dir, item)
            dst = os.path.join(output_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

        return output_dir
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def upload_file_to_s3(s3, file_path, bucket, key, progress_cb=None):
    content_type = content_type_for(os.path.basename(file_path))

    for attempt in range(UPLOAD_MAX_RETRIES):
        try:
            with open(file_path, 'rb') as f:
                s3.upload_fileobj(f, bucket, key, ExtraArgs={'ContentType': content_type})
            return
        except Exception as e:
            if attempt < UPLOAD_MAX_RETRIES - 1:
                if progress_cb:
                    progress_cb(f"Upload failed, retrying ({attempt + 1}/{UPLOAD_MAX_RETRIES})")
                time.sleep(UPLOAD_RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"Failed to upload {os.path.basename(file_path)} after {UPLOAD_MAX_RETRIES} attempts: {e}"
                ) from e


def encode_image_tensor_to_webp(image_tensor, output_path, quality=90):
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_np)
    image.save(output_path, format='WEBP', quality=quality, method=4)


def encode_image_tensor_to_webp_bytes(image_tensor, quality=85):
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_np)
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', quality=quality, method=4)
    buffer.seek(0)
    return buffer
