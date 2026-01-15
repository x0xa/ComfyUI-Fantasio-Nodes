import io
import uuid
import boto3
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from server import PromptServer


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
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "fantasio"

    def process(self, images, quality=85, thumb_quality=75, thumb_size=600,
                s3_endpoint="", s3_access_key="", s3_secret_key="",
                s3_bucket="", s3_public_url="", client_id=""):

        if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket, s3_public_url]):
            raise ValueError("S3 credentials missing")

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
                    s3_bucket, s3_public_url, client_id
                )
                for idx, image_tensor in enumerate(images)
            ]
            for future in futures:
                future.result()

        return {"ui": {"images": []}}

    def _process_single_image(self, s3, image_tensor, idx, quality, thumb_quality, thumb_size, bucket, public_url, client_id):
        sid = client_id if client_id else None

        h, w = image_tensor.shape[:2]
        filename = str(uuid.uuid4())
        orientation = self._get_orientation(w, h)
        main_key = f"generated/originals/{orientation}/{filename}.webp"
        thumb_key = f"generated/thumbnails/{orientation}/{filename}_thumb.webp"

        # Calculate thumbnail dimensions
        if w > h:
            thumb_w, thumb_h = thumb_size, int(h * thumb_size / w)
        else:
            thumb_h, thumb_w = thumb_size, int(w * thumb_size / h)

        # Resize thumbnail on GPU (much faster than CPU)
        # tensor shape: (H, W, C) -> (1, C, H, W) for F.interpolate
        tensor_for_resize = image_tensor.permute(2, 0, 1).unsqueeze(0)
        thumb_tensor = F.interpolate(tensor_for_resize, size=(thumb_h, thumb_w), mode='bilinear', align_corners=False)
        thumb_tensor = thumb_tensor.squeeze(0).permute(1, 2, 0)

        # Convert both to numpy/PIL (move to CPU here)
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        thumb_np = (thumb_tensor.cpu().numpy() * 255).astype(np.uint8)

        img = Image.fromarray(img_np)
        thumb = Image.fromarray(thumb_np)

        # Encode main and thumbnail in parallel threads
        def encode_main():
            buf = io.BytesIO()
            img.save(buf, format='WEBP', quality=quality, method=2)
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

        # Upload both files in parallel
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

    def _get_orientation(self, w, h):
        if w > h:
            return "landscape"
        elif h > w:
            return "portrait"
        return "square"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {"SaveWebPToS3": SaveWebPToS3}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveWebPToS3": "Save WebP to S3"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
