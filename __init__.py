import io
import os
import time
import uuid
import threading
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from server import PromptServer
from nodes import VAEEncode, VAEDecode, CLIPTextEncode
import folder_paths

from .lib import create_s3_client, normalize_s3_public_url, download_to_file, download_and_extract_archive, describe_exception

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


def _send_error(error_message, node_name, sid=None):
    payload = {
        "error": str(error_message),
        "node": node_name,
    }

    PromptServer.instance.send_sync("fa.node.error", payload, sid)


def _download_progress_cb(sid):
    return lambda percent: _send_gpu_activity(
        f"Downloading: {percent}%", sid=sid, value=percent, max_value=100
    )


class GpuActivityNotifier:
    def __init__(self, message, sid, interval=None):
        self.message = message
        self.sid = sid
        self.interval = interval if interval is not None else _GPU_ACTIVITY_INTERVAL_SECONDS
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        _send_gpu_activity(self.message, sid=self.sid)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        return False

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            _send_gpu_activity(self.message, sid=self.sid)


def _build_thumb_image(image_tensor, w, h, thumb_size):
    if w > h:
        thumb_w, thumb_h = thumb_size, int(h * thumb_size / w)
    else:
        thumb_h, thumb_w = thumb_size, int(w * thumb_size / h)

    tensor_for_resize = image_tensor.permute(2, 0, 1).unsqueeze(0)
    thumb_tensor = F.interpolate(tensor_for_resize, size=(thumb_h, thumb_w), mode='bilinear', align_corners=False)
    thumb_tensor = thumb_tensor.squeeze(0).permute(1, 2, 0)
    thumb_np = (thumb_tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(thumb_np)


def _encode_webp(image, quality, method):
    buf = io.BytesIO()
    try:
        image.save(buf, format='WEBP', quality=quality, method=method)
        return buf.getvalue()
    finally:
        buf.close()


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
                "convert": ("BOOLEAN", {"default": True}),
                "upload": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "s3_endpoint": ("STRING",),
                "s3_access_key": ("STRING",),
                "s3_secret_key": ("STRING",),
                "s3_bucket": ("STRING",),
                "s3_public_url": ("STRING",),
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "fantasio"

    def process(self, images, quality=85, thumb_quality=75, thumb_size=600,
                s3_endpoint="", s3_access_key="", s3_secret_key="",
                s3_bucket="", s3_public_url="", client_id="",
                convert=True, upload=True):
        sid = client_id if client_id else None
        do_upload = convert and upload

        try:
            if do_upload:
                missing = [name for name, value in (
                    ("endpoint", s3_endpoint),
                    ("access_key", s3_access_key),
                    ("secret_key", s3_secret_key),
                    ("bucket", s3_bucket),
                    ("public_url", s3_public_url),
                ) if not value]
                if missing:
                    raise ValueError(f"S3 credentials missing: {', '.join(missing)}")

            return self._process_images(images, quality, thumb_quality, thumb_size,
                                        s3_endpoint, s3_access_key, s3_secret_key,
                                        s3_bucket, s3_public_url, sid, convert, do_upload)
        except Exception as e:
            _send_error(describe_exception(e), "SaveWebPToS3", sid=sid)
            raise

    def _process_images(self, images, quality, thumb_quality, thumb_size,
                        s3_endpoint, s3_access_key, s3_secret_key,
                        s3_bucket, s3_public_url, sid, convert, do_upload):

        s3 = create_s3_client(s3_endpoint, s3_access_key, s3_secret_key) if do_upload else None

        with GpuActivityNotifier("Saving images to S3" if do_upload else "Saving images", sid):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        self._process_single_image,
                        s3, image_tensor, idx,
                        quality, thumb_quality, thumb_size,
                        s3_bucket, s3_public_url, sid, convert, do_upload
                    )
                    for idx, image_tensor in enumerate(images)
                ]
                for future in futures:
                    future.result()

        return {"ui": {"images": []}}

    def _process_single_image(self, s3, image_tensor, idx, quality, thumb_quality, thumb_size, bucket, public_url, sid, convert, do_upload):
        h, w = image_tensor.shape[:2]
        filename = str(uuid.uuid4())
        orientation = self._get_orientation(w, h)

        if not do_upload:
            self._save_local_image(image_tensor, w, h, orientation, filename, quality, thumb_quality, thumb_size, convert, sid)
            return

        main_key = f"generated/originals/{orientation}/{filename}.webp"
        thumb_key = f"generated/thumbnails/{orientation}/{filename}_thumb.webp"

        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        thumb = _build_thumb_image(image_tensor, w, h, thumb_size)

        with ThreadPoolExecutor(max_workers=2) as enc_executor:
            main_future = enc_executor.submit(_encode_webp, img, quality, 4)
            thumb_future = enc_executor.submit(_encode_webp, thumb, thumb_quality, 2)
            main_bytes = main_future.result()
            thumb_bytes = thumb_future.result()

        def upload(key, body):
            s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/webp')

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with ThreadPoolExecutor(max_workers=2) as upload_executor:
                    main_upload_future = upload_executor.submit(upload, main_key, main_bytes)
                    thumb_upload_future = upload_executor.submit(upload, thumb_key, thumb_bytes)
                    main_upload_future.result()
                    thumb_upload_future.result()

                main_url = normalize_s3_public_url(public_url, main_key)
                thumb_url = normalize_s3_public_url(public_url, thumb_key)

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
                    continue

                size_bytes = len(main_bytes) + len(thumb_bytes)
                detail = (
                    f"S3 upload failed for {main_key} (+thumbnail) in bucket '{bucket}', "
                    f"{size_bytes} bytes, after {max_retries} attempts: {describe_exception(e)}"
                )
                PromptServer.instance.send_sync("s3-upload-failed", {
                    "error": detail,
                    "index": idx,
                    "bucket": bucket,
                    "key": main_key,
                    "thumb_key": thumb_key,
                    "attempts": max_retries,
                    "size_bytes": size_bytes,
                }, sid)
                raise e

    def _save_local_image(self, image_tensor, w, h, orientation, filename, quality, thumb_quality, thumb_size, convert, sid):
        output_dir = folder_paths.get_output_directory()
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        if convert:
            saved_filename = f"{filename}.webp"
            img.save(os.path.join(output_dir, saved_filename), format='WEBP', quality=quality, method=4)
            image_format = "webp"
        else:
            saved_filename = f"{filename}.png"
            img.save(os.path.join(output_dir, saved_filename), format='PNG')
            image_format = "png"

        payload = {
            "filename": saved_filename,
            "orientation": orientation,
            "width": w,
            "height": h,
            "format": image_format,
        }

        if convert:
            thumb = _build_thumb_image(image_tensor, w, h, thumb_size)
            thumb_filename = f"{filename}_thumb.webp"
            thumb.save(os.path.join(output_dir, thumb_filename), format='WEBP', quality=thumb_quality, method=2)
            payload["thumb_filename"] = thumb_filename

        PromptServer.instance.send_sync("gpu-image-saved", payload, sid)

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
                "deadline_seconds": ("INT", {"default": 0, "min": 0, "max": 3600}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "return_basename": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING", "MODEL", "CLIP")
    RETURN_NAMES = ("output_path", "model", "clip")
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, output_path, model=None, clip=None, timeout_seconds=60, deadline_seconds=0, overwrite=False, return_basename=False, client_id=""):
        sid = client_id if client_id else None

        try:
            if not overwrite and os.path.isfile(output_path):
                _send_gpu_activity(f"File already exists: {output_path}", sid=sid)
                return ((os.path.basename(output_path) if return_basename else output_path), model, clip)

            with GpuActivityNotifier(f"Downloading file to {output_path}", sid):
                download_to_file(url, output_path, timeout_seconds=timeout_seconds, progress_cb=_download_progress_cb(sid), deadline_seconds=(deadline_seconds or None))
            _send_gpu_activity(f"Downloaded file to {output_path}", sid=sid)

            return ((os.path.basename(output_path) if return_basename else output_path), model, clip)
        except Exception as e:
            _send_error(str(e), "FantasioDownloadFile", sid=sid)
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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "run"
    CATEGORY = "loaders"

    def run(self, model, clip, lora_name, strength_model, strength_clip, client_id=""):
        sid = client_id if client_id else None

        try:
            if strength_model == 0 and strength_clip == 0:
                return (model, clip)

            import folder_paths
            import comfy.sd
            import comfy.utils

            lora_path = None
            searched_paths = []

            if os.path.isabs(lora_name):
                lora_path = lora_name
                searched_paths.append(lora_name)
            else:
                env_lora_dir = os.environ.get("COMFY_LORAS_DIR", "").strip()
                if env_lora_dir:
                    candidate = os.path.join(env_lora_dir, lora_name)
                    searched_paths.append(candidate)
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
                        searched_paths.append(candidate)
                        if os.path.isfile(candidate):
                            lora_path = candidate
                            break

            if lora_path is None or not os.path.isfile(lora_path):
                searched = ", ".join(searched_paths) if searched_paths else "no candidate paths"
                raise RuntimeError(f"LoRA file not found: {lora_name} (searched: {searched})")

            with GpuActivityNotifier(f"Applying LoRA {os.path.basename(lora_path)}", sid):
                try:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to read LoRA file {lora_path}: {describe_exception(e)}") from e

                try:
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to apply LoRA {os.path.basename(lora_path)} "
                        f"(strength_model={strength_model}, strength_clip={strength_clip}): {describe_exception(e)}"
                    ) from e

            return (model_lora, clip_lora)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioLoraLoader", sid=sid)
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, timeout_seconds=60, client_id=""):
        sid = client_id if client_id else None

        try:
            with GpuActivityNotifier("Downloading input image", sid):
                request = urllib.request.Request(url, headers={"User-Agent": "fantasio-comfy-node/1.0"})

                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    data = response.read()

                try:
                    image = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception as e:
                    raise RuntimeError(
                        f"Downloaded {len(data)} bytes from {url} but could not decode as an image "
                        f"(content may be an error page or unsupported format): {describe_exception(e)}"
                    ) from e

            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)

            _send_gpu_activity("Input image loaded", sid=sid)

            return (image_tensor.unsqueeze(0),)
        except Exception as e:
            _send_error(f"Failed to load image from {url}: {describe_exception(e)}", "FantasioLoadImageFromUrl", sid=sid)
            raise


class FantasioDownloadAndExtractArchive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
                "output_dir": ("STRING", {"multiline": False}),
            },
            "optional": {
                "archive_name": ("STRING", {"default": ""}),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 3600}),
                "clean_archive": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder_path",)
    FUNCTION = "run"
    CATEGORY = "fantasio/io"

    def run(self, url, output_dir, archive_name="", timeout_seconds=300, clean_archive=True, client_id=""):
        sid = client_id if client_id else None

        try:
            with GpuActivityNotifier("Downloading dataset archive", sid):
                download_and_extract_archive(
                    url,
                    output_dir,
                    archive_name=archive_name,
                    timeout_seconds=timeout_seconds,
                    progress_cb=_download_progress_cb(sid),
                )
            _send_gpu_activity(f"Dataset extracted to {output_dir}", sid=sid)

            return (output_dir,)
        except Exception as e:
            _send_error(str(e), "FantasioDownloadAndExtractArchive", sid=sid)
            raise


class FantasioApplyTriggerWord:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"forceInput": True}),
                "trigger_word": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "fantasio"

    def run(self, caption, trigger_word=""):
        text = caption.strip() if isinstance(caption, str) else ""
        word = trigger_word.strip() if isinstance(trigger_word, str) else ""

        if not word:
            return (text,)

        if not text:
            return (word,)

        return (f"{word}, {text}",)


class FantasioSampleCaptionEmitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = "fantasio"

    def run(self, caption, client_id=None):
        sid = client_id[0] if isinstance(client_id, list) and client_id else client_id
        captions = caption if isinstance(caption, list) else [caption]
        first = next((c.strip() for c in captions if isinstance(c, str) and c.strip()), "")

        if first and sid:
            PromptServer.instance.send_sync("training.caption.sample", {"caption": first}, sid)

        return {"ui": {}}


class FantasioUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": ("STRING", {"multiline": False}),
            },
            "optional": {
                "weight_dtype": ("STRING", {"default": "default"}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "fantasio/loaders"

    def run(self, unet_name, weight_dtype="default", client_id=""):
        sid = client_id if client_id else None

        try:
            import folder_paths
            import comfy.sd

            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            unet_basename = os.path.basename(unet_path)

            with GpuActivityNotifier(f"Loading diffusion model {unet_basename}", sid):
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            _send_gpu_activity(f"Diffusion model {unet_basename} loaded", sid=sid)

            return (model,)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioUNETLoader", sid=sid)
            raise


class FantasioCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": ("STRING", {"multiline": False}),
                "type": ("STRING", {"default": "stable_diffusion"}),
            },
            "optional": {
                "device": ("STRING", {"default": "default"}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "run"
    CATEGORY = "fantasio/loaders"

    def run(self, clip_name, type="stable_diffusion", device="default", client_id=""):
        sid = client_id if client_id else None

        try:
            import folder_paths
            import comfy.sd

            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

            model_options = {}
            if device == "cpu":
                model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            clip_basename = os.path.basename(clip_path)

            with GpuActivityNotifier(f"Loading text encoder {clip_basename}", sid):
                clip = comfy.sd.load_clip(
                    ckpt_paths=[clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options=model_options,
                )
            _send_gpu_activity(f"Text encoder {clip_basename} loaded", sid=sid)

            return (clip,)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioCLIPLoader", sid=sid)
            raise


class FantasioVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": ("STRING", {"multiline": False}),
            },
            "hidden": {
                "client_id": ("STRING",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"
    CATEGORY = "fantasio/loaders"

    def run(self, vae_name, client_id=""):
        sid = client_id if client_id else None

        try:
            import folder_paths
            import comfy.sd
            import comfy.utils

            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae_basename = os.path.basename(vae_path)

            with GpuActivityNotifier(f"Loading VAE {vae_basename}", sid):
                sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
                vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                vae.throw_exception_if_invalid()
            _send_gpu_activity(f"VAE {vae_basename} loaded", sid=sid)

            return (vae,)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioVAELoader", sid=sid)
            raise


class FantasioVAEEncode(VAEEncode):
    @classmethod
    def INPUT_TYPES(cls):
        types = VAEEncode.INPUT_TYPES()
        types.setdefault("hidden", {})["client_id"] = ("STRING",)
        return types

    FUNCTION = "run"
    CATEGORY = "fantasio/latent"

    def run(self, pixels, vae, client_id=""):
        sid = client_id if client_id else None
        try:
            with GpuActivityNotifier("Encoding image to latent", sid):
                return super().encode(vae, pixels)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioVAEEncode", sid=sid)
            raise


class FantasioVAEDecode(VAEDecode):
    @classmethod
    def INPUT_TYPES(cls):
        types = VAEDecode.INPUT_TYPES()
        types.setdefault("hidden", {})["client_id"] = ("STRING",)
        return types

    FUNCTION = "run"
    CATEGORY = "fantasio/latent"

    def run(self, samples, vae, client_id=""):
        sid = client_id if client_id else None
        try:
            with GpuActivityNotifier("Decoding latent to image", sid):
                return super().decode(vae, samples)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioVAEDecode", sid=sid)
            raise


class FantasioCLIPTextEncode(CLIPTextEncode):
    @classmethod
    def INPUT_TYPES(cls):
        types = CLIPTextEncode.INPUT_TYPES()
        types.setdefault("hidden", {})["client_id"] = ("STRING",)
        return types

    FUNCTION = "run"
    CATEGORY = "fantasio/conditioning"

    def run(self, text, clip, client_id=""):
        sid = client_id if client_id else None
        try:
            with GpuActivityNotifier("Encoding prompt (text encoder)", sid):
                return super().encode(clip, text)
        except Exception as e:
            _send_error(describe_exception(e), "FantasioCLIPTextEncode", sid=sid)
            raise


NODE_CLASS_MAPPINGS = {
    "SaveWebPToS3": SaveWebPToS3,
    "FantasioVAEEncode": FantasioVAEEncode,
    "FantasioVAEDecode": FantasioVAEDecode,
    "FantasioCLIPTextEncode": FantasioCLIPTextEncode,
    "FantasioDownloadFile": FantasioDownloadFile,
    "FantasioLoraLoader": FantasioLoraLoader,
    "FantasioLoadImageFromUrl": FantasioLoadImageFromUrl,
    "FantasioDownloadAndExtractArchive": FantasioDownloadAndExtractArchive,
    "FantasioApplyTriggerWord": FantasioApplyTriggerWord,
    "FantasioSampleCaptionEmitter": FantasioSampleCaptionEmitter,
    "FantasioUNETLoader": FantasioUNETLoader,
    "FantasioCLIPLoader": FantasioCLIPLoader,
    "FantasioVAELoader": FantasioVAELoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPToS3": "Save WebP to S3",
    "FantasioVAEEncode": "Fantasio VAE Encode",
    "FantasioVAEDecode": "Fantasio VAE Decode",
    "FantasioCLIPTextEncode": "Fantasio CLIP Text Encode",
    "FantasioDownloadFile": "Fantasio Download File",
    "FantasioLoraLoader": "Fantasio LoRA Loader",
    "FantasioLoadImageFromUrl": "Fantasio Load Image From URL",
    "FantasioDownloadAndExtractArchive": "Fantasio Download And Extract Archive",
    "FantasioApplyTriggerWord": "Fantasio Apply Trigger Word",
    "FantasioSampleCaptionEmitter": "Fantasio Sample Caption Emitter",
    "FantasioUNETLoader": "Fantasio UNET Loader",
    "FantasioCLIPLoader": "Fantasio CLIP Loader",
    "FantasioVAELoader": "Fantasio VAE Loader",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
