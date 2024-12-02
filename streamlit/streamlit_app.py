import os
import sys
if "APP_PATH" in os.environ:
    os.chdir(os.environ["APP_PATH"])
    # fix sys.path for import
    sys.path.append(os.getcwd())

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from st_keyup import st_keyup

import hashlib

import string
import random
import time
import re
import pandas as pd

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torchvision import transforms

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Proportion of the image to be watermarked (0.5 means 50% of the image).
# This is used here to show the watermark localization property. In practice, you may want to use a predifined mask or the entire image.
proportion_masked = 0.5

# DBSCAN parameters for detection
epsilon = 1 # min distance between decoded messages in a cluster
min_samples = 500 # min number of pixels in a 256x256 image to form a cluster

# Define a color map for each unique value for multiple wm viz
color_map = {
    -1: [0, 0, 0],      # Black for -1
    0: [255, 0, 255],   # ? for 0
    1: [255, 0, 0],     # Red for 1
    2: [0, 255, 0],     # Green for 2
    3: [0, 0, 255],     # Blue for 3
    4: [255, 255, 0],   # Yellow for 4
}

@st.cache_resource()
def load_wam():
    # Load the model from the specified checkpoint
    exp_dir = "checkpoints"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    return wam


def image_detect(img_pil: Image.Image):
    img_pt = default_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Detect the watermark in the multi-watermarked image
    preds = wam.detect(img_pt)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

    # positions has the cluster number at each pixel. can be upsaled back to the original size.
    try:
        centroids, positions = multiwm_dbscan(bit_preds, mask_preds, epsilon = epsilon, min_samples = min_samples)
        centroids_pt = torch.stack(list(centroids.values()))
    except (UnboundLocalError) as e:
        print(f"Error while detecting watermark: {e}")
        positions = None
        centroids = None
        centroids_pt = None

    return img_pt, (mask_preds_res>0.5).float(), positions, centroids, centroids_pt

def image_embed(img_pil: Image.Image, wm_msgs: torch.Tensor, wm_masks: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    img_pt = default_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Embed the watermark message into the image
    # Mask to use. 1 values correspond to pixels where the watermark will be placed.
    multi_wm_img = img_pt.clone()
    for ii in range(len(wm_msgs)):
        wm_msg, mask = wm_msgs[ii].unsqueeze(0), wm_masks[ii]
        outputs = wam.embed(img_pt, wm_msg)
        multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)  # [1, 3, H, W]

    torch.cuda.empty_cache()
    return img_pt, multi_wm_img, wm_masks.sum(0)

def centroid_to_hex(centroid) -> str:
    """Convert a 32-bit binary array to 8-character hex string.

    Args:
        centroid: A tensor representing 32-bit binary array

    Returns:
        An 8-character hex string
    """
    # Convert binary array to integer
    binary_int = 0
    for bit in centroid:
        binary_int = (binary_int << 1) | int(bit.item())
    # Convert to 8-character hex string (32 bits = 8 hex chars)
    return format(binary_int, '08x')

def get_canvas_hash(img_pil):
    return hashlib.md5(img_pil.tobytes()).hexdigest()

@st.cache_data()
def get_image_size(img_pil):
    if img_pil is None:
        return MAX_HEIGHT, MAX_WIDTH
    height, width = img_pil.height, img_pil.width
    return height, width


st.set_page_config(layout="wide")

st.markdown("""
# Watermark Anything Demo

This app will let you try watermark anything model.

Find the project [here](https://github.com/facebookresearch/watermark-anything).
""")

col1, col2, col3 = st.columns([.3, .3, .3])
col0, = st.columns([1])


wam = load_wam()


in_file = st.sidebar.file_uploader("image file:", type=["png", "jpg", "jpeg", "gif", "webp"])
if in_file is None:
    st.stop()

img_pil = Image.open(in_file).convert("RGB")

if img_pil is None:
    st.stop()


detecting_btn = st.sidebar.button("Run detect")
# Run detect
if detecting_btn:
    torch.cuda.empty_cache()

    det_img, pred, positions, centroids, centroids_pt = image_detect(img_pil)

    with col1:
        np_det = torch_to_np(det_img.detach())
        st.image(np_det, caption="Detected image", use_container_width=True)

    with col2:
        np_pred = torch_to_np(pred.detach().repeat(1, 3, 1, 1))
        st.image(np_pred, caption="Predicted watermark position", use_container_width=True)

    with col3:
        if positions is None:
            st.warning(f"No watermarks detected.")
            st.stop()

        resize_ori = transforms.Resize(det_img.shape[-2:])
        full_labels_store = positions
        rgb_image = torch.zeros((3, full_labels_store.shape[-1], full_labels_store.shape[-2]), dtype=torch.uint8)
        # Map each value to its corresponding color
        for value, color in color_map.items():
            mask_ = full_labels_store == value
            for channel, color_value in enumerate(color):
                rgb_image[channel][mask_.squeeze()] = color_value
        rgb_image = resize_ori(rgb_image.float()/255)
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
        st.image(rgb_image, caption="clusters", use_container_width=True)

    with col0:
        st.markdown(f"Number messages found in image: {len(centroids)}")
        for key in centroids.keys():
            centroid_hex = centroid_to_hex(centroids[key])
            centroid_hex_array = "-".join([centroid_hex[i:i+4] for i in range(0, len(centroid_hex), 4)])
            st.markdown(f'<code style="color:rgb{tuple(color_map[key])}">{centroid_hex_array}</code>', unsafe_allow_html=True)

    st.stop()

embedding_num = st.sidebar.slider(f"Number of watermarks:", min_value=1, max_value=5, value=1)

embedding_type = st.sidebar.radio("Type of watermarks:", ["random", "input"], index=0)
if embedding_type == "input":
    tip = "-".join([f"FFFF-FFFF" for _ in range(embedding_num)])
    with st.sidebar:
        # Create a keyup component
        embedding_str = st_keyup(f"Input hex: like {tip}", key="0")

    if not re.match(r"^([0-9A-F]{4}-[0-9A-F]{4}-){%d}[0-9A-F]{4}-[0-9A-F]{4}$" % (embedding_num-1), embedding_str):
        st.sidebar.error(f"Invalid format. Please use {tip}")

embedding_loc = st.sidebar.radio("Location of watermarks:", ["random", "bounding"], index=0)
if embedding_loc == "random":
    with col1:
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
else:
    canvas_hash = get_canvas_hash(img_pil) if img_pil else "canvas"
    with col1:
        w = 300
        scale = w / get_image_size(img_pil)[1]
        h = int(get_image_size(img_pil)[0] * scale)
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#FFAA00",
            background_color="#FFF",
            background_image=img_pil,
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="rect",
            point_display_radius=0,
            key=canvas_hash
        )

        objects = None
        if canvas_result.json_data:
            objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        if objects is not None and objects.shape[0] > 0:
            boxes = objects[objects["type"] == "rect"][["left", "top", "width", "height"]]
            boxes["right"] = boxes["left"] + boxes["width"]
            boxes["bottom"] = boxes["top"] + boxes["height"]
            bbox_list = (boxes[["left", "top", "right", "bottom"]].values / scale).astype(int).tolist()
        else:
            bbox_list = [(0, 0, img_pil.width, img_pil.height)]
            st.info(f"No bounding boxes found. Using full image.")

        if len(bbox_list) != embedding_num:
            st.warning(f"Number of bounding ({len(bbox_list)}) does not match the number of watermarks ({embedding_num}).")
            st.stop()


embedding_btn = st.sidebar.button("Run embed")
# Run embed
if embedding_btn:
    torch.cuda.empty_cache()

    wm_msgs = []
    if embedding_type == "random":
        chars = '-'.join(''.join(random.choice(string.hexdigits) for _ in range(4)) for _ in range(embedding_num * 2))
        embedding_str = chars.lower()
    with col0:
        st.markdown(f"Watermark code: `{embedding_str}`")
    wm_hex = embedding_str.replace("-", "")
    for i in range(0, len(wm_hex), 8):
        chunk = wm_hex[i:i+8]
        binary = bin(int(chunk, 16))[2:].zfill(32)
        wm_msgs.append([int(b) for b in binary])
    # Define a 32-bit message to be embedded into the images
    wm_msgs = torch.tensor(wm_msgs, dtype=torch.float32).to(device)


    wm_masks = None
    if embedding_loc == "random":
        img_pt = default_transform(img_pil).unsqueeze(0).to(device)
        # To ensure at least `proportion_masked %` of the width is randomly usable,
        # otherwise, it is easy to enter an infinite loop and fail to find a usable width.
        mask_percentage = img_pil.height / img_pil.width * proportion_masked / wm_num
        wm_masks = create_random_mask(img_pt, num_masks=embedding_num, mask_percentage=mask_percentage)
    elif embedding_loc == "bounding" and bbox_list:
        wm_masks = torch.zeros((len(bbox_list), 1, img_pil.height, img_pil.width), dtype=torch.float32).to(device)
        for idx, (left, top, right, bottom) in enumerate(bbox_list):
            wm_masks[idx, 0, top:bottom, left:right] = 1


    img_pt, embed_img_pt, embed_mask_pt = image_embed(img_pil, wm_msgs, wm_masks)

    with col2:
        embed_img_np = torch_to_np(embed_img_pt.detach())
        st.image(embed_img_np, caption="Watermarked image", use_container_width=True)
    with col3:
        embed_mask_np = torch_to_np(embed_mask_pt.detach().expand(3, -1, -1))
        st.image(embed_mask_np, caption="Position of the watermark", use_container_width=True)
