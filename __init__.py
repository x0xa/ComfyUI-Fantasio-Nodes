import io
import os
import time
import uuid
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from server import PromptServer

from .lib import create_s3_client, normalize_s3_public_url, download_to_file

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


def _download_progress_cb(sid):
    return lambda percent: _send_gpu_activity(
        f"Downloading: {percent}%", sid=sid, value=percent, max_value=100
    )


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

        s3 = create_s3_client(s3_endpoint, s3_access_key, s3_secret_key)

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
                s3.upload_fileobj(main_buffer, bucket, main_key, ExtraArgs={'ContentType': 'image/webp'})

            def upload_thumb():
                s3.upload_fileobj(thumb_buffer, bucket, thumb_key, ExtraArgs={'ContentType': 'image/webp'})

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with ThreadPoolExecutor(max_workers=2) as upload_executor:
                        main_upload_future = upload_executor.submit(upload_main)
                        thumb_upload_future = upload_executor.submit(upload_thumb)
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
            download_to_file(url, output_path, timeout_seconds=timeout_seconds, progress_cb=_download_progress_cb(sid))
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


NODE_CLASS_MAPPINGS = {
    "SaveWebPToS3": SaveWebPToS3,
    "FantasioDownloadFile": FantasioDownloadFile,
    "FantasioLoraLoader": FantasioLoraLoader,
    "FantasioLoadImageFromUrl": FantasioLoadImageFromUrl,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWebPToS3": "Save WebP to S3",
    "FantasioDownloadFile": "Fantasio Download File",
    "FantasioLoraLoader": "Fantasio LoRA Loader",
    "FantasioLoadImageFromUrl": "Fantasio Load Image From URL",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
