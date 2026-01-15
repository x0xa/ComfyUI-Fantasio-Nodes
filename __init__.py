import io
import uuid
import boto3
import numpy as np
from PIL import Image
from botocore.config import Config
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

        for idx, image_tensor in enumerate(images):
            self._process_single_image(
                s3, image_tensor, idx,
                quality, thumb_quality, thumb_size,
                s3_bucket, s3_public_url, client_id
            )

        return {"ui": {"images": []}}

    def _process_single_image(self, s3, image_tensor, idx, quality, thumb_quality,
                               thumb_size, bucket, public_url, client_id):
        sid = client_id if client_id else None

        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        filename = str(uuid.uuid4())
        orientation = self._get_orientation(img.width, img.height)

        main_buffer = io.BytesIO()
        img.save(main_buffer, format='WEBP', quality=quality, method=4)
        main_key = f"generated/originals/{orientation}/{filename}.webp"

        thumb = self._create_thumbnail(img, thumb_size)
        thumb_buffer = io.BytesIO()
        thumb.save(thumb_buffer, format='WEBP', quality=thumb_quality, method=4)
        thumb_key = f"generated/thumbnails/{orientation}/{filename}_thumb.webp"

        main_uploaded = False
        thumb_uploaded = False
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if not main_uploaded:
                    main_buffer.seek(0)
                    s3.upload_fileobj(
                        main_buffer, bucket, main_key,
                        ExtraArgs={'ContentType': 'image/webp'}
                    )
                    main_uploaded = True

                if not thumb_uploaded:
                    thumb_buffer.seek(0)
                    s3.upload_fileobj(
                        thumb_buffer, bucket, thumb_key,
                        ExtraArgs={'ContentType': 'image/webp'}
                    )
                    thumb_uploaded = True

                main_url = f"{public_url.rstrip('/')}/{main_key}"
                thumb_url = f"{public_url.rstrip('/')}/{thumb_key}"

                PromptServer.instance.send_sync("s3-image-uploaded", {
                    "url": main_url,
                    "thumb_url": thumb_url,
                    "path": main_key,
                    "thumb_path": thumb_key,
                    "orientation": orientation,
                    "width": img.width,
                    "height": img.height,
                }, sid)

                return

            except Exception as e:
                if attempt < max_retries - 1:
                    continue

                PromptServer.instance.send_sync("s3-upload-failed", {
                    "error": str(e),
                    "index": idx,
                }, sid)
                raise e

    def _create_thumbnail(self, img, size):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = img.crop((left, top, left + min_dim, top + min_dim))
        return cropped.resize((size, size), Image.LANCZOS)

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
NODE_DISPLAY_NAME_MAPPINGS = {"SaveWebPToS3": "Save WebP to S3 (Fantasio)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
