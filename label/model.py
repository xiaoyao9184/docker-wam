import os
import sys
if "APP_PATH" in os.environ:
    os.chdir(os.environ["APP_PATH"])
    # fix sys.path for import
    sys.path.append(os.getcwd())

import json
import boto3
import logging
import re
import string
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from uuid import uuid4

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint,
    default_transform,
    unnormalize_img,
    create_random_mask,
    plot_outputs,
    msg2str,
    torch_to_np,
    multiwm_dbscan
)

from typing import List, Dict, Optional
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
proportion_masked = 0.5  # Proportion of image to be watermarked
epsilon = 1  # min distance between decoded messages in a cluster
min_samples = 500  # min number of pixels in a 256x256 image to form a cluster

def load_wam():
    # Load the model from the specified checkpoint
    exp_dir = "checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    return wam

# Load the model
wam = load_wam()

class WAM(LabelStudioMLBase):
    """Custom ML Backend model
    """
    # Label Studio image upload folder:
    # should be used only in case you use direct file upload into Label Studio instead of URLs
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def _get_image_url(self, task, value):
        # TODO: warning! currently only s3 presigned urls are supported with the default keys
        # also it seems not be compatible with file directly uploaded to Label Studio
        # check RND-2 for more details and fix it later
        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)

        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def image_detect(self, img_pil: Image.Image) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        img_pt = default_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Detect the watermark in the multi-watermarked image
        preds = wam.detect(img_pt)["preds"]  # [1, 33, 256, 256]
        mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
        mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
        bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

        # positions has the cluster number at each pixel. can be upsaled back to the original size.
        try:
            centroids, positions = multiwm_dbscan(bit_preds, mask_preds, epsilon=epsilon, min_samples=min_samples)
            centroids_pt = torch.stack(list(centroids.values()))
        except (UnboundLocalError) as e:
            print(f"Error while detecting watermark: {e}")
            positions = None
            centroids = None
            centroids_pt = None

        return img_pt, (mask_preds_res>0.5).float(), positions, centroids, centroids_pt

    def centroid_to_hex(self, centroid):
        binary_int = 0
        for bit in centroid:
            binary_int = (binary_int << 1) | int(bit.item())
        return format(binary_int, '08x')

    def predict_single(self, task):
        logger.debug('Task data: %s', task['data'])
        from_name_area, to_name, value = self.get_first_tag_occurence('TextArea', 'Image', name_filter=lambda x: x == 'watermark_msg')
        from_name_brush, _, _ = self.get_first_tag_occurence('BrushLabels', 'Image', name_filter=lambda x: x == 'watermark_mask')

        labels = []
        for idx, l in enumerate(self.label_interface.labels):
            if 'watermarked' in l.keys() and l['watermarked'].parent_name == 'watermark_mask':
                labels.append(l)
        if len(labels) != 1:
            logger.error("More than one 'watermarked' label in the tag.")
        label = 'watermarked'

        image_url = self._get_image_url(task, value)
        cache_dir = os.path.join(self.MODEL_DIR, '.file-cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f'Using cache dir: {cache_dir}')
        image_path = get_local_path(
            image_url,
            cache_dir=cache_dir,
            hostname=self.LABEL_STUDIO_HOST,
            access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
            task_id=task.get('id')
        )

        # run detect
        img_pil = Image.open(image_path).convert("RGB")
        det_img, pred, positions, centroids, centroids_pt = self.image_detect(img_pil)

        if positions is None:
            return

        result = []
        all_scores = []
        img_width, img_height = img_pil.width, img_pil.height

        for key in centroids.keys():
            id_gen = str(uuid4())[:4]

            # no score just allway 1
            score = 1

            # mask 2 rle
            pred_res = F.interpolate(positions.unsqueeze(0).unsqueeze(0), size=det_img.shape[-2:], mode="nearest")  # [1, 1, H, W]
            mask_bool = pred_res == key
            mask_int = mask_bool.squeeze().squeeze().numpy().astype(np.uint8)
            rle = brush.mask2rle(mask_int * 255)

            centroid_hex = self.centroid_to_hex(centroids[key])
            centroid_hex_array = "-".join([centroid_hex[i:i+4] for i in range(0, len(centroid_hex), 4)])

            # must add one for the Brush
            result.append({
                'id': id_gen,
                'from_name': from_name_brush,
                'to_name': to_name,
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    "format": "rle",
                    "rle": rle,
                    'brushlabels': [label]
                },
                'score': score,
                'type': 'brushlabels',
                'readonly': False
            })

            # and one for the TextArea
            result.append({
                'id': id_gen,
                'from_name': from_name_area,
                'to_name': to_name,
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    "text": [centroid_hex_array],
                    'brushlabels': [label]
                },
                'score': score,
                'type': 'textarea',
                'origin': 'manual'
            })
            all_scores.append(score)
        return {
            'result': result,
            'score': sum(all_scores) / max(len(all_scores), 1),
            'model_version': self.get('model_version'),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        for task in tasks:
            # TODO: implement is_skipped() function
            # if is_skipped(task):
            #     continue

            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        return ModelResponse(predictions=predictions, model_versions=self.get('model_version'))
