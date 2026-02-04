import io
import json
import os
import time
import uuid
import tarfile
import zipfile
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
import boto3
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from server import PromptServer

_GPU_ACTIVITY_LAST_SENT_AT = {}
_GPU_ACTIVITY_INTERVAL_SECONDS = max(0.5, float(os.environ.get("FANTASIO_GPU_ACTIVITY_INTERVAL_SECONDS", "5.0")))


def _send_gpu_activity(message, sid=None, value=None, max_value=None, node=None):
    if sid is not None and value is not None and max_value is not None:
        is_terminal_progress = float(value) >= float(max_value)
        if not is_terminal_progress:
            now = time.monotonic()
            last_sent_at = _GPU_ACTIVITY_LAST_SENT_AT.get(sid, 0.0)
            if now - last_sent_at < _GPU_ACTIVITY_INTERVAL_SECONDS:
                return
            _GPU_ACTIVITY_LAST_SENT_AT[sid] = now

    payload = {"message": message}

    if value is not None:
        payload["value"] = value

    if max_value is not None:
        payload["max"] = max_value

    if node is not None:
        payload["node"] = node

    PromptServer.instance.send_sync("fa.gpu.activity", payload, sid)


def _send_error(error_message, node_name, task_id=None, sid=None):
    payload = {
        "error": str(error_message),
        "node": node_name,
    }
    if task_id is not None:
        payload["task_id"] = int(task_id)

    PromptServer.instance.send_sync("fa.node.error", payload, sid)


def _download_to_file(url, output_path, timeout_seconds=60, chunk_size=1024 * 512, sid=None):
    import socket
    from urllib.error import URLError, HTTPError

    request = urllib.request.Request(url, headers={"User-Agent": "fantasio-comfy-node/1.0"})

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total else None

            downloaded = 0
            progress_interval_seconds = max(0.5, float(os.environ.get("FANTASIO_DOWNLOAD_PROGRESS_INTERVAL_SECONDS", "5.0")))
            last_reported_percent = 0
            last_reported_at = 0.0

            with open(output_path, "wb") as output_file:
                while True:
                    chunk = response.read(chunk_size)

                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes:
                        percent = int((downloaded / total_bytes) * 100)
                        if percent >= 100:
                            continue

                        now = time.monotonic()
                        if percent > last_reported_percent and now - last_reported_at >= progress_interval_seconds:
                            _send_gpu_activity(f"Downloading: {percent}%", sid=sid, value=percent, max_value=100)
                            last_reported_percent = percent
                            last_reported_at = now

            _send_gpu_activity("Downloading: 100%", sid=sid, value=100, max_value=100)

    except HTTPError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"HTTP error downloading {url}: {e.code} {e.reason}") from e
    except URLError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"URL error downloading {url}: {e.reason}") from e
    except socket.timeout:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Timeout downloading {url} after {timeout_seconds}s")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Failed to download {url}: {e}") from e


class SaveWebPToS3:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "quality": ("INT", {"default": 85, "min": 1, "max": 100}),
                "thumb_quality": ("INT", {"default": 75, "min": 1, "max": 100}),
                "thumb_size": ("INT", {"default": 600, "min": 100, "max": 1200}),
            },
            "hidden": {
                "s3_endpoint": ("STRING",),
                "s3_access_key": ("STRING",),
                "s3_secret_key": ("STRING",),
                "s3_bucket": ("STRING",),
                "s3_public_url": ("STRING",),
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "fantasio"

    def process(self, images, quality=85, thumb_quality=75, thumb_size=600,
                s3_endpoint="", s3_access_key="", s3_secret_key="",
                s3_bucket="", s3_public_url="", client_id="", task_id=0):
        sid = client_id if client_id else None

        try:
            if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket, s3_public_url]):
                raise ValueError("S3 credentials missing")

            return self._process_images(images, quality, thumb_quality, thumb_size,
                                        s3_endpoint, s3_access_key, s3_secret_key,
                                        s3_bucket, s3_public_url, sid, task_id)
        except Exception as e:
            _send_error(str(e), "SaveWebPToS3", task_id=task_id, sid=sid)
            raise

    def _process_images(self, images, quality, thumb_quality, thumb_size,
                        s3_endpoint, s3_access_key, s3_secret_key,
                        s3_bucket, s3_public_url, sid, task_id):

        s3 = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._process_single_image,
                    s3, image_tensor, idx,
                    quality, thumb_quality, thumb_size,
                    s3_bucket, s3_public_url, sid
                )
                for idx, image_tensor in enumerate(images)
            ]
            for future in futures:
                future.result()

        return {"ui": {"images": []}}

    def _process_single_image(self, s3, image_tensor, idx, quality, thumb_quality, thumb_size, bucket, public_url, sid):
        main_buffer = None
        thumb_buffer = None

        try:
            h, w = image_tensor.shape[:2]
            filename = str(uuid.uuid4())
            orientation = self._get_orientation(w, h)
            main_key = f"generated/originals/{orientation}/{filename}.webp"
            thumb_key = f"generated/thumbnails/{orientation}/{filename}_thumb.webp"

            if w > h:
                thumb_w, thumb_h = thumb_size, int(h * thumb_size / w)
            else:
                thumb_h, thumb_w = thumb_size, int(w * thumb_size / h)

            tensor_for_resize = image_tensor.permute(2, 0, 1).unsqueeze(0)
            thumb_tensor = F.interpolate(tensor_for_resize, size=(thumb_h, thumb_w), mode='bilinear', align_corners=False)
            thumb_tensor = thumb_tensor.squeeze(0).permute(1, 2, 0)

            img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            thumb_np = (thumb_tensor.cpu().numpy() * 255).astype(np.uint8)

            img = Image.fromarray(img_np)
            thumb = Image.fromarray(thumb_np)

            def encode_main():
                buf = io.BytesIO()
                img.save(buf, format='WEBP', quality=quality, method=4)
                buf.seek(0)
                return buf

            def encode_thumb():
                buf = io.BytesIO()
                thumb.save(buf, format='WEBP', quality=thumb_quality, method=2)
                buf.seek(0)
                return buf

            with ThreadPoolExecutor(max_workers=2) as enc_executor:
                main_future = enc_executor.submit(encode_main)
                thumb_future = enc_executor.submit(encode_thumb)
                main_buffer = main_future.result()
                thumb_buffer = thumb_future.result()

            def upload_main():
                s3.upload_fileobj(
                    main_buffer, bucket, main_key,
                    ExtraArgs={'ContentType': 'image/webp'}
                )

            def upload_thumb():
                s3.upload_fileobj(
                    thumb_buffer, bucket, thumb_key,
                    ExtraArgs={'ContentType': 'image/webp'}
                )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with ThreadPoolExecutor(max_workers=2) as upload_executor:
                        main_upload_future = upload_executor.submit(upload_main)
                        thumb_upload_future = upload_executor.submit(upload_thumb)
                        main_upload_future.result()
                        thumb_upload_future.result()

                    main_url = f"{public_url.rstrip('/')}/{main_key}"
                    thumb_url = f"{public_url.rstrip('/')}/{thumb_key}"

                    PromptServer.instance.send_sync("s3-image-uploaded", {
                        "url": main_url,
                        "thumb_url": thumb_url,
                        "path": main_key,
                        "thumb_path": thumb_key,
                        "orientation": orientation,
                        "width": w,
                        "height": h,
                    }, sid)

                    return

                except Exception as e:
                    if attempt < max_retries - 1:
                        main_buffer.seek(0)
                        thumb_buffer.seek(0)
                        continue

                    PromptServer.instance.send_sync("s3-upload-failed", {
                        "error": str(e),
                        "index": idx,
                    }, sid)
                    raise e
        finally:
            if main_buffer:
                main_buffer.close()
            if thumb_buffer:
                thumb_buffer.close()

    def _get_orientation(self, w, h):
        if w > h:
            return "landscape"
        elif h > w:
            return "portrait"
        return "square"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class FantasioDownloadFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
                "output_path": ("STRING", {"multiline": False}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 3600}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "return_basename": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL", "CLIP")
    RETURN_NAMES = ("output_path", "model", "clip")
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, output_path, model=None, clip=None, timeout_seconds=60, overwrite=False, return_basename=False, client_id="", task_id=0):
        sid = client_id if client_id else None

        try:
            if not overwrite and os.path.isfile(output_path):
                _send_gpu_activity(f"File already exists: {output_path}", sid=sid)
                return ((os.path.basename(output_path) if return_basename else output_path), model, clip)

            _send_gpu_activity(f"Downloading file to {output_path}", sid=sid)
            _download_to_file(url, output_path, timeout_seconds=timeout_seconds, sid=sid)
            _send_gpu_activity(f"Downloaded file to {output_path}", sid=sid)

            return ((os.path.basename(output_path) if return_basename else output_path), model, clip)
        except Exception as e:
            _send_error(str(e), "FantasioDownloadFile", task_id=task_id, sid=sid)
            raise


class FantasioLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING", {"multiline": False}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "run"
    CATEGORY = "loaders"

    def run(self, model, clip, lora_name, strength_model, strength_clip, client_id="", task_id=0):
        sid = client_id if client_id else None

        try:
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)

            import folder_paths
            import comfy.sd
            import comfy.utils

            lora_path = None

            if os.path.isabs(lora_name):
                lora_path = lora_name
            else:
                env_lora_dir = os.environ.get("COMFY_LORAS_DIR", "").strip()
                if env_lora_dir:
                    candidate = os.path.join(env_lora_dir, lora_name)
                    if os.path.isfile(candidate):
                        lora_path = candidate

                if lora_path is None:
                    full_path = folder_paths.get_full_path("loras", lora_name)
                    if full_path and os.path.isfile(full_path):
                        lora_path = full_path

                if lora_path is None:
                    lora_folders = folder_paths.get_folder_paths("loras")
                    for folder in lora_folders:
                        candidate = os.path.join(folder, lora_name)
                        if os.path.isfile(candidate):
                            lora_path = candidate
                            break

            if lora_path is None or not os.path.isfile(lora_path):
                raise RuntimeError(f"LoRA file not found: {lora_name}")

            _send_gpu_activity(f"Loading LoRA {os.path.basename(lora_path)}", sid=sid)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

            return (model_lora, clip_lora)
        except Exception as e:
            _send_error(str(e), "FantasioLoraLoader", task_id=task_id, sid=sid)
            raise


def _validate_archive_member(member_name, output_dir):
    if os.path.isabs(member_name):
        raise ValueError(f"Path traversal detected: absolute path '{member_name}'")
    if '..' in member_name.split(os.sep) or '..' in member_name.split('/'):
        raise ValueError(f"Path traversal detected: '..' in path '{member_name}'")
    resolved = os.path.normpath(os.path.join(output_dir, member_name))
    if not resolved.startswith(os.path.normpath(output_dir)):
        raise ValueError(f"Path traversal detected: '{member_name}' escapes output directory")


class FantasioDownloadAndExtractArchive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
                "output_dir": ("STRING", {"multiline": False}),
            },
            "optional": {
                "archive_name": ("STRING", {"multiline": False, "default": ""}),
                "timeout_seconds": ("INT", {"default": 120, "min": 5, "max": 7200}),
                "clean_archive": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, output_dir, archive_name="", timeout_seconds=120, clean_archive=True, client_id="", task_id=0):
        import shutil
        import tempfile

        sid = client_id if client_id else None
        temp_dir = None
        archive_path = None

        try:
            temp_dir = tempfile.mkdtemp(prefix="fantasio_archive_")

            if archive_name.strip():
                filename = archive_name.strip()
            else:
                parsed = urllib.parse.urlparse(url)
                filename = os.path.basename(parsed.path) or f"archive_{uuid.uuid4().hex}.zip"

            archive_path = os.path.join(temp_dir, filename)

            _send_gpu_activity(f"Downloading archive to {archive_path}", sid=sid)
            _download_to_file(url, archive_path, timeout_seconds=timeout_seconds, sid=sid)

            lower_name = archive_path.lower()
            _send_gpu_activity(f"Extracting archive {filename}", sid=sid)

            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)

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

            os.makedirs(output_dir, exist_ok=True)
            for item in os.listdir(extract_dir):
                src = os.path.join(extract_dir, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                else:
                    shutil.move(src, dst)

            shutil.rmtree(temp_dir, ignore_errors=True)

            _send_gpu_activity(f"Archive extracted to {output_dir}", sid=sid)

            return (output_dir,)
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            _send_error(str(e), "FantasioDownloadAndExtractArchive", task_id=task_id, sid=sid)
            raise


class FantasioLoadImageFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
            },
            "optional": {
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 3600}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, timeout_seconds=60, client_id="", task_id=0):
        sid = client_id if client_id else None

        try:
            _send_gpu_activity("Downloading input image", sid=sid)

            request = urllib.request.Request(url, headers={"User-Agent": "fantasio-comfy-node/1.0"})

            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = response.read()

            image = Image.open(io.BytesIO(data)).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)

            _send_gpu_activity("Input image loaded", sid=sid)

            return (image_tensor.unsqueeze(0),)
        except Exception as e:
            _send_error(str(e), "FantasioLoadImageFromUrl", task_id=task_id, sid=sid)
            raise



def _create_s3_client(s3_endpoint, s3_access_key, s3_secret_key):
    return boto3.client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )


def _normalize_s3_public_url(base_url, key):
    return f"{base_url.rstrip('/')}/{key.lstrip('/')}"


class FantasioConvertImagesToWebP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"multiline": False}),
            },
            "optional": {
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "prefix": ("STRING", {"default": "sample"}),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, images, output_dir, quality=90, prefix="sample", client_id="", task_id=0):
        import shutil
        import tempfile

        sid = client_id if client_id else None
        temp_dir = None

        try:
            os.makedirs(output_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="fantasio_webp_")

            temp_paths = []
            final_paths = []

            total = len(images)

            for index, image_tensor in enumerate(images):
                filename = f"{prefix}_{index}_{uuid.uuid4().hex}.webp"
                temp_path = os.path.join(temp_dir, filename)
                final_path = os.path.join(output_dir, filename)

                img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(temp_path, format='WEBP', quality=quality, method=4)

                temp_paths.append(temp_path)
                final_paths.append(final_path)

                _send_gpu_activity(
                    f"Converted samples to webp: {index + 1}/{total}",
                    sid=sid,
                    value=index + 1,
                    max_value=total,
                )

            for temp_path, final_path in zip(temp_paths, final_paths):
                shutil.move(temp_path, final_path)

            shutil.rmtree(temp_dir, ignore_errors=True)

            return ("\n".join(final_paths),)
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            _send_error(str(e), "FantasioConvertImagesToWebP", task_id=task_id, sid=sid)
            raise


class FantasioUploadLocalFileToS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"multiline": False}),
                "key_prefix": ("STRING", {"multiline": False}),
            },
            "optional": {
                "delete_after_upload": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "s3_endpoint": ("STRING",),
                "s3_access_key": ("STRING",),
                "s3_secret_key": ("STRING",),
                "s3_bucket": ("STRING",),
                "s3_public_url": ("STRING",),
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("url", "key", "file_size")
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(
        self,
        file_path,
        key_prefix,
        delete_after_upload=False,
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
        s3_bucket="",
        s3_public_url="",
        client_id="",
        task_id=0,
    ):
        import time

        sid = client_id if client_id else None
        max_retries = 3
        retry_delay = 3

        try:
            if not os.path.isfile(file_path):
                raise ValueError(f"File not found: {file_path}")

            if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket, s3_public_url]):
                raise ValueError("S3 credentials missing")

            filename = os.path.basename(file_path)
            key = f"{key_prefix.rstrip('/')}/{filename}"
            file_size = os.path.getsize(file_path)

            _send_gpu_activity(f"Uploading file to s3: {filename}", sid=sid)

            s3 = _create_s3_client(s3_endpoint, s3_access_key, s3_secret_key)
            content_type = 'image/webp' if filename.lower().endswith('.webp') else 'application/octet-stream'

            for attempt in range(max_retries):
                try:
                    with open(file_path, 'rb') as f:
                        s3.upload_fileobj(f, s3_bucket, key, ExtraArgs={'ContentType': content_type})
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        _send_gpu_activity(f"Upload failed, retrying ({attempt + 1}/{max_retries})...", sid=sid)
                        time.sleep(retry_delay)
                    else:
                        raise RuntimeError(f"Failed to upload {filename} after {max_retries} attempts: {e}") from e

            if delete_after_upload and os.path.exists(file_path):
                os.remove(file_path)

            url = _normalize_s3_public_url(s3_public_url, key)
            _send_gpu_activity(f"Uploaded file to s3: {filename}", sid=sid)

            return (url, key, file_size)
        except Exception as e:
            _send_error(str(e), "FantasioUploadLocalFileToS3", task_id=task_id, sid=sid)
            raise


class FantasioUploadLocalFilesToS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_paths": ("STRING", {"multiline": True}),
                "key_prefix": ("STRING", {"multiline": False}),
            },
            "optional": {
                "delete_after_upload": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "s3_endpoint": ("STRING",),
                "s3_access_key": ("STRING",),
                "s3_secret_key": ("STRING",),
                "s3_bucket": ("STRING",),
                "s3_public_url": ("STRING",),
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("urls",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(
        self,
        file_paths,
        key_prefix,
        delete_after_upload=False,
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
        s3_bucket="",
        s3_public_url="",
        client_id="",
        task_id=0,
    ):
        import time

        sid = client_id if client_id else None

        try:
            if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket, s3_public_url]):
                raise ValueError("S3 credentials missing")

            paths = [line.strip() for line in file_paths.splitlines() if line.strip()]
            if not paths:
                return ("",)

            for path in paths:
                if not os.path.isfile(path):
                    raise ValueError(f"File not found: {path}")

            s3 = _create_s3_client(s3_endpoint, s3_access_key, s3_secret_key)
            uploaded_keys = []
            total = len(paths)
            max_retries = 3
            retry_delay = 3

            try:
                for index, path in enumerate(paths):
                    filename = os.path.basename(path)
                    key = f"{key_prefix.rstrip('/')}/{filename}"

                    for attempt in range(max_retries):
                        try:
                            with open(path, 'rb') as f:
                                s3.upload_fileobj(f, s3_bucket, key, ExtraArgs={'ContentType': 'image/webp'})
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                _send_gpu_activity(f"Upload failed, retrying ({attempt + 1}/{max_retries})...", sid=sid)
                                time.sleep(retry_delay)
                            else:
                                raise RuntimeError(f"Failed to upload {filename} after {max_retries} attempts: {e}") from e

                    uploaded_keys.append((path, key))

                    _send_gpu_activity(
                        f"Uploaded sample {index + 1}/{total}",
                        sid=sid,
                        value=index + 1,
                        max_value=total,
                    )

            except Exception:
                for _, key in uploaded_keys:
                    try:
                        s3.delete_object(Bucket=s3_bucket, Key=key)
                    except:
                        pass
                raise

            if delete_after_upload:
                for path, _ in uploaded_keys:
                    if os.path.exists(path):
                        os.remove(path)

            urls = [_normalize_s3_public_url(s3_public_url, key) for _, key in uploaded_keys]
            return ("\n".join(urls),)
        except Exception as e:
            _send_error(str(e), "FantasioUploadLocalFilesToS3", task_id=task_id, sid=sid)
            raise


class FantasioExtractTrainingMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
            },
            "hidden": {
                "client_id": ("STRING",),
                "task_id": ("INT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("avg_loss", "step", "epoch")
    FUNCTION = "run"
    CATEGORY = "fantasio/training"

    def run(self, network_trainer, client_id="", task_id=0):
        sid = client_id if client_id else None

        try:
            trainer = network_trainer["network_trainer"]
            avg_loss = float(getattr(trainer.loss_recorder, 'moving_average', 0.0) or 0.0)
            step = int(getattr(trainer, 'global_step', 0) or 0)
            epoch = int(getattr(trainer.current_epoch, 'value', 0) or 0)
            return (avg_loss, step, epoch)
        except Exception as e:
            _send_error(str(e), "FantasioExtractTrainingMetrics", task_id=task_id, sid=sid)
            raise


class FantasioEmitTrainingEpochUploaded:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_trainer": ("NETWORKTRAINER",),
                "task_id": ("INT", {"default": 0, "min": 0}),
                "user_id": ("INT", {"default": 0, "min": 0}),
                "total_epochs": ("INT", {"default": 0, "min": 0}),
                "epoch": ("INT", {"default": 0, "min": 0}),
                "avg_loss": ("FLOAT", {"default": 0.0}),
                "step": ("INT", {"default": 0, "min": 0}),
                "lora_url": ("STRING", {"multiline": False}),
                "sample_urls": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("NETWORKTRAINER",)
    RETURN_NAMES = ("network_trainer",)
    FUNCTION = "run"
    CATEGORY = "fantasio/training"

    def run(self, network_trainer, task_id, user_id, total_epochs, epoch, avg_loss, step, lora_url, sample_urls, client_id=""):
        sid = client_id if client_id else None

        try:
            urls = [line.strip() for line in sample_urls.splitlines() if line.strip()]
            payload = {
                "task_id": int(task_id),
                "user_id": int(user_id),
                "epoch": int(epoch),
                "avg_loss": float(avg_loss),
                "step": int(step),
                "lora_url": lora_url,
                "sample_images": [{"url": url} for url in urls],
            }

            PromptServer.instance.send_sync("training.epoch.uploaded", payload, sid)

            if int(total_epochs) > 0 and int(epoch) >= int(total_epochs):
                selection_expires_at = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat().replace('+00:00', 'Z')
                PromptServer.instance.send_sync("training.task.completed", {
                    "task_id": int(task_id),
                    "user_id": int(user_id),
                    "status": "TrainingCompleted",
                    "epoch_number": int(epoch),
                    "selection_expires_at": selection_expires_at,
                }, sid)

            _send_gpu_activity(f"Epoch {epoch} artifacts uploaded", sid=sid)

            return (network_trainer,)
        except Exception as e:
            _send_error(str(e), "FantasioEmitTrainingEpochUploaded", task_id=task_id, sid=sid)
            raise

NODE_CLASS_MAPPINGS = {
    "SaveWebPToS3": SaveWebPToS3,
    "FantasioDownloadFile": FantasioDownloadFile,
    "FantasioLoraLoader": FantasioLoraLoader,
    "FantasioDownloadAndExtractArchive": FantasioDownloadAndExtractArchive,
    "FantasioLoadImageFromUrl": FantasioLoadImageFromUrl,
    "FantasioConvertImagesToWebP": FantasioConvertImagesToWebP,
    "FantasioUploadLocalFileToS3": FantasioUploadLocalFileToS3,
    "FantasioUploadLocalFilesToS3": FantasioUploadLocalFilesToS3,
    "FantasioExtractTrainingMetrics": FantasioExtractTrainingMetrics,
    "FantasioEmitTrainingEpochUploaded": FantasioEmitTrainingEpochUploaded,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPToS3": "Save WebP to S3",
    "FantasioDownloadFile": "Fantasio Download File",
    "FantasioLoraLoader": "Fantasio LoRA Loader",
    "FantasioDownloadAndExtractArchive": "Fantasio Download And Extract Archive",
    "FantasioLoadImageFromUrl": "Fantasio Load Image From URL",
    "FantasioConvertImagesToWebP": "Fantasio Convert Images To WebP",
    "FantasioUploadLocalFileToS3": "Fantasio Upload Local File To S3",
    "FantasioUploadLocalFilesToS3": "Fantasio Upload Local Files To S3",
    "FantasioExtractTrainingMetrics": "Fantasio Extract Training Metrics",
    "FantasioEmitTrainingEpochUploaded": "Fantasio Emit Training Epoch Uploaded",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
