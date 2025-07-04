import os
import sys
if "APP_PATH" in os.environ:
    # fix sys.path for import
    os.chdir(os.environ["APP_PATH"])
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

# remove duplicate gradio_app path from sys.path
sys.path = list(dict.fromkeys(sys.path))

# remove gradio reload env if in huggingface space
if "SPACE_ID" in os.environ:
    for key in ["GRADIO_WATCH_DIRS", "GRADIO_WATCH_MODULE_NAME", "GRADIO_WATCH_DEMO_NAME", "GRADIO_WATCH_DEMO_PATH"]:
        if key in os.environ:
            del os.environ[key]

# here the subprocess stops loading, because __name__ is NOT '__main__'
# gradio will reload
if '__main__' == __name__:

    import gradio as gr
    import requests
    from contextlib import suppress

    import os
    import re
    import string
    import random
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    from watermark_anything.data.metrics import msg_predict_inference
    from notebooks.inference_utils import (
        load_model_from_checkpoint,
        default_transform,
        create_random_mask,
        torch_to_np
    )

    import time
    from multiwm import dbscan

    max_timeout = int(os.environ.get("MAX_TIMEOUT", 60))

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = 42
    torch.manual_seed(seed)

    # Constants
    proportion_masked = 0.5  # Proportion of image to be watermarked
    epsilon = 1  # min distance between decoded messages in a cluster
    min_samples = 500  # min number of pixels in a 256x256 image to form a cluster

    # Color map for visualization
    color_map = {
        -1: [0, 0, 0],      # Black for -1
        0: [255, 0, 255],   # ? for 0
        1: [255, 0, 0],     # Red for 1
        2: [0, 255, 0],     # Green for 2
        3: [0, 0, 255],     # Blue for 3
        4: [255, 255, 0],   # Yellow for 4
        5: [0, 255, 255],   # ?
    }

    def load_wam():
        # Load the model from the specified checkpoint
        exp_dir = "checkpoints"
        json_path = os.path.join(exp_dir, "params.json")
        ckpt_path = os.environ.get("CHECKPOINT_MODEL_PATH", exp_dir)
        ckpt_file = os.path.join(ckpt_path, 'checkpoint.pth')
        wam = load_model_from_checkpoint(json_path, ckpt_file).to(device).eval()
        return wam

    def image_detect(img_pil: Image.Image, scan_mult: bool=False, timeout_seconds: int=5) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        img_pt = default_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Detect the watermark in the multi-watermarked image
        preds = wam.detect(img_pt)["preds"]  # [1, 33, 256, 256]
        mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
        bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

        mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
        message_pred_inf = msg_predict_inference(bit_preds, mask_preds).cpu().float()  # [1, 32]
        if message_pred_inf.sum() == 0:
            message_pred_inf = None

        centroids, positions = None, None
        if(scan_mult):
            try:
                centroids, positions = dbscan(bit_preds, mask_preds, epsilon, min_samples, dec_timeout=timeout_seconds)
            except TimeoutError:
                print("Timeout error in multiwm task!")
            except (UnboundLocalError) as e:
                print(f"Error while detecting watermark: {e}")

        return img_pt, (mask_preds_res>0.5).float(), message_pred_inf, positions, centroids

    def image_embed(img_pil: Image.Image, wm_msgs: torch.Tensor, wm_masks: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        img_pt = default_transform(img_pil).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Embed the watermark message into the image
        # Mask to use. 1 values correspond to pixels where the watermark will be placed.
        multi_wm_img = img_pt.clone()
        for ii in range(len(wm_msgs)):
            wm_msg, mask = wm_msgs[ii].unsqueeze(0), wm_masks[ii]
            outputs = wam.embed(img_pt, wm_msg)
            multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)

        return img_pt, multi_wm_img, wm_masks.sum(0)

    def create_bounding_mask(img_size, boxes):
        """Create a binary mask from bounding boxes.

        Args:
            img_size (tuple): Image size (height, width)
            boxes (list): List of tuples (x1, y1, x2, y2) defining bounding boxes

        Returns:
            torch.Tensor: Binary mask tensor
        """
        mask = torch.zeros(img_size)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 1
        return mask

    def centroid_to_hex(centroid):
        binary_int = 0
        for bit in centroid:
            binary_int = (binary_int << 1) | int(bit.item())
        return format(binary_int, '08x')

    # Load the model
    wam = load_wam()

    def detect_watermark(image, multi, timeout):
        """
        Detects watermark(s) in the given image.

        Args:
            image (Union[PIL.Image.Image, str]): Input image, either a PIL Image object or a URL pointing to the image.
            multi (bool): If True, enables detection of multiple watermarks in the image.
            timeout (float): Timeout in seconds for image loading or processing; used to handle unresponsive or invalid images.

        Returns:
            tuple: A 4-element tuple containing:
                - dict: An image dict of the predicted watermark positions.
                - dict: An image dict of watermark clusters.
                - Union[str, float, bool, list, dict]: Metadata or messages related to the detection.
                - str: A markdown string summarizing the detection result (e.g., confidence level).
        """
        # update counter
        with suppress(Exception):
            requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-wam")

        if image is None:
            return None, None, None, {"status": "error", "messages": [], "error": "No image provided"}

        start_time = time.time()

        img_pil = Image.fromarray(image).convert("RGB")
        det_img, mask_preds_res, message_pred_inf, positions, centroids = image_detect(img_pil, multi, timeout)

        # Convert tensor images to numpy for display
        pred_mask = torch_to_np(mask_preds_res.detach().repeat(1, 3, 1, 1))

        cluster_viz = None
        message_json = {
            "status": "none-detected"
        }

        if message_pred_inf is not None:
            cluster_viz = pred_mask

            centroid_hex = centroid_to_hex(message_pred_inf[0])
            centroid_hex_array = "-".join([centroid_hex[i:i+4] for i in range(0, len(centroid_hex), 4)])
            message_json['status'] = "one-detected"
            message_json['message'] = centroid_hex_array

        # Create cluster visualization
        if positions is not None:
            resize_ori = transforms.Resize(det_img.shape[-2:])
            rgb_image = torch.zeros((3, positions.shape[-1], positions.shape[-2]), dtype=torch.uint8)
            for value, color in color_map.items():
                mask_ = positions == value
                for channel, color_value in enumerate(color):
                    rgb_image[channel][mask_.squeeze()] = color_value
            rgb_image = resize_ori(rgb_image.float()/255)
            cluster_viz = rgb_image.permute(1, 2, 0).numpy()

            # Create message output as JSON
            messages = []
            for key in centroids.keys():
                centroid_hex = centroid_to_hex(centroids[key])
                centroid_hex_array = "-".join([centroid_hex[i:i+4] for i in range(0, len(centroid_hex), 4)])
                messages.append({
                    "id": int(key),
                    "message": centroid_hex_array,
                    "color": color_map[key]
                })
            message_json['status'] = "multi-detected"
            message_json['cluster'] = messages

        run_time = time.time() - start_time
        message_json['run_time'] = run_time

        color_md = []
        if "cluster" in message_json:
            for item in message_json["cluster"]:
                key = item["id"]
                msg = item["message"]
                color_md.append(f'<code style="color:rgb{tuple(color_map[key])}">{msg}</code>')

        return pred_mask, cluster_viz, message_json, "\n".join(color_md)

    def embed_watermark(image, wm_num, wm_type, wm_str, wm_loc):
        """
        Embeds one or more watermarks into the input image.

        Args:
            image (Union[PIL.Image.Image, str]): The input image, either as a PIL Image object or a URL string.
            wm_num (float): The number of watermarks to embed.
            wm_type (Literal['random', 'input']): The type of watermark. 
                - 'random': Use randomly generated watermark content.
                - 'input': Use the content from `wm_str` as the watermark.
            wm_str (str): The text content of the watermark when `wm_type='input'`.
            wm_loc (Literal['random', 'bounding']): The placement mode of the watermark.
                - 'random': Randomly place the watermark(s).
                - 'bounding': Place within a bounding box or defined area.

        Returns:
            tuple: A 3-element tuple containing:
                - dict: Watermarked image output (shown in "Watermarked Image" component).
                - dict: Image showing the watermark position (shown in "Position of the watermark").
                - Union[str, float, bool, list, dict]: Metadata or message related to the embedding process (shown in "Marked Messages").
        """
        # update counter
        with suppress(Exception):
            requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/docker-wam")

        if image is None:
            return None, None, {
                "status": "failure",
                "messages": "No image provided"
            }

        if wm_type == "input":
            if not re.match(r"^([0-9A-F]{4}-[0-9A-F]{4}-){%d}[0-9A-F]{4}-[0-9A-F]{4}$" % (wm_num-1), wm_str):
                tip = "-".join([f"FFFF-{_}{_}{_}{_}" for _ in range(wm_num)])
                return None, None, {
                    "status": "failure",
                    "messages": f"Invalid type input. Please use {tip}"
                }

        if wm_loc == "bounding":
            if ROI_coordinates['clicks'] != wm_num * 2:
                return None, None, {
                    "status": "failure",
                    "messages": "Invalid location input. Please draw at least %d bounding ROI" % (wm_num)
                }

        img_pil = Image.fromarray(image).convert("RGB")

        # Generate watermark messages based on type
        wm_msgs = []
        if wm_type == "random":
            chars = '-'.join(''.join(random.choice(string.hexdigits) for _ in range(4)) for _ in range(wm_num * 2))
            wm_str = chars.lower()
        wm_hex = wm_str.replace("-", "")
        for i in range(0, len(wm_hex), 8):
            chunk = wm_hex[i:i+8]
            binary = bin(int(chunk, 16))[2:].zfill(32)
            wm_msgs.append([int(b) for b in binary])
        # Define a 32-bit message to be embedded into the images
        wm_msgs = torch.tensor(wm_msgs, dtype=torch.float32).to(device)

        # Create mask based on location type
        wm_masks = None
        if wm_loc == "random":
            img_pt = default_transform(img_pil).unsqueeze(0).to(device)
            # To ensure at least `proportion_masked %` of the width is randomly usable,
            # otherwise, it is easy to enter an infinite loop and fail to find a usable width.
            mask_percentage = min(img_pil.height, img_pil.width) / max(img_pil.height, img_pil.width) * proportion_masked / wm_num
            wm_masks = create_random_mask(img_pt, num_masks=wm_num, mask_percentage=mask_percentage)
        elif wm_loc == "bounding" and sections:
            wm_masks = torch.zeros((len(sections), 1, img_pil.height, img_pil.width), dtype=torch.float32).to(device)
            for idx, ((x_start, y_start, x_end, y_end), _) in enumerate(sections):
                left = min(x_start, x_end)
                right = max(x_start, x_end)
                top = min(y_start, y_end)
                bottom = max(y_start, y_end)
                wm_masks[idx, 0, top:bottom, left:right] = 1


        img_pt, embed_img_pt, embed_mask_pt = image_embed(img_pil, wm_msgs, wm_masks)

        # Convert to numpy for display
        img_np = torch_to_np(embed_img_pt.detach())
        mask_np = torch_to_np(embed_mask_pt.detach().expand(3, -1, -1))
        message_json = {
            "status": "success",
            "messages": wm_str
        }
        return img_np, mask_np, message_json



    # ROI means Region Of Interest. It is the region where the user clicks
    # to specify the location of the watermark.
    ROI_coordinates = {
        'x_temp': 0,
        'y_temp': 0,
        'x_new': 0,
        'y_new': 0,
        'clicks': 0,
    }

    sections = []

    def get_select_coordinates(img, evt: gr.SelectData, num):
        if ROI_coordinates['clicks'] >= num * 2:
            gr.Warning(f"Cant add more than {num} of Watermarks.")
            return (img, sections)

        # update new coordinates
        ROI_coordinates['clicks'] += 1
        ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
        ROI_coordinates['y_temp'] = ROI_coordinates['y_new']
        ROI_coordinates['x_new'] = evt.index[0]
        ROI_coordinates['y_new'] = evt.index[1]
        # compare start end coordinates
        x_start = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_start = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
        x_end = ROI_coordinates['x_new'] if (ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
        y_end = ROI_coordinates['y_new'] if (ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
        if ROI_coordinates['clicks'] % 2 == 0:
            sections[len(sections) - 1] = ((x_start, y_start, x_end, y_end), f"Mask {len(sections)}")
            # both start and end point get
            return (img, sections)
        else:
            point_width = int(img.shape[0]*0.05)
            sections.append(((ROI_coordinates['x_new'], ROI_coordinates['y_new'],
                            ROI_coordinates['x_new'] + point_width, ROI_coordinates['y_new'] + point_width),
                            f"Click second point for Mask {len(sections) + 1}"))
            return (img, sections)

    def del_select_coordinates(img, evt: gr.SelectData):
        del sections[evt.index]
        # recreate section names
        for i in range(len(sections)):
            sections[i] = (sections[i][0], f"Mask {i + 1}")

        # last section clicking second point not complete
        if ROI_coordinates['clicks'] % 2 != 0:
            if len(sections) == evt.index:
                # delete last section
                ROI_coordinates['clicks'] -= 1
            else:
                # recreate last section name for second point
                ROI_coordinates['clicks'] -= 2
                sections[len(sections) - 1] = (sections[len(sections) - 1][0], f"Click second point for Mask {len(sections) + 1}")
        else:
            ROI_coordinates['clicks'] -= 2

        return (img[0], sections)

    with gr.Blocks(title="Watermark Anything Demo") as demo:
        gr.Markdown("""
        # Watermark Anything Demo
        ![](https://badge.mcpx.dev?type=server 'MCP Server')
        This app demonstrates watermark detection and embedding using the Watermark Anything model.
        
        Find the original project [here](https://github.com/facebookresearch/watermark-anything).
        Or this project [here](https://github.com/xiaoyao9184/docker-wam).
        See the [README](./blob/main/README.md) for Spaces's metadata.
        """)

        with gr.Tabs():
            with gr.TabItem("Embed Watermark"):
                with gr.Row():
                    with gr.Column():
                        embedding_img = gr.Image(label="Input Image", type="numpy")

                    with gr.Column():
                        embedding_box = gr.AnnotatedImage(
                            visible=False,
                            label="ROI: Click on left 'Input Image'",
                            color_map={
                                "ROI of Watermark embedding": "#9987FF",
                                "Click second point for ROI": "#f44336"}
                        )

                        embedding_num = gr.Slider(1, 5, value=1, step=1, label="Number of Watermarks")
                        embedding_type = gr.Radio(["random", "input"], value="random", label="Type", info="Type of watermarks")
                        embedding_str = gr.Textbox(label="Watermark Text", visible=False, show_copy_button=True)
                        embedding_loc = gr.Radio(["random", "bounding"], value="random", label="Location", info="Location of watermarks")

                        embedding_btn = gr.Button("Embed Watermark")
                        marked_msg = gr.JSON(label="Marked Messages")
                with gr.Row():
                    marked_image = gr.Image(label="Watermarked Image")
                    marked_mask = gr.Image(label="Position of the watermark")

                embedding_img.select(
                    fn=get_select_coordinates,
                    inputs=[embedding_img, embedding_num],
                    outputs=embedding_box,
                    api_name=False)
                embedding_box.select(
                    fn=del_select_coordinates,
                    inputs=embedding_box,
                    outputs=embedding_box,
                    api_name=False
                )

                # The inability to dynamically render `AnnotatedImage` is because,
                # when placed inside `gr.Column()`, it prevents listeners from being added to controls outside the column.
                # Dynamically adding a select listener will not change the cursor shape of the Image.
                # So `render` cannot work properly in this scenario.
                #
                # @gr.render(inputs=embedding_loc)
                # def show_split(wm_loc):
                #     if wm_loc == "bounding":
                #         embedding_img.select(
                #             fn=get_select_coordinates,
                #             inputs=[embedding_img, embedding_num],
                #             outputs=embedding_box)
                #         embedding_box.select(
                #             fn=del_select_coordinates,
                #             inputs=embedding_box,
                #             outputs=embedding_box
                #         )
                #     else:
                #         embedding_img.select()

                def visible_box_image(img, wm_loc):
                    if wm_loc == "bounding":
                        return gr.update(visible=True, value=(img,sections))
                    else:
                        sections.clear()
                        ROI_coordinates['clicks'] = 0
                        return gr.update(visible=False, value=(img,sections))
                embedding_loc.change(
                    fn=visible_box_image,
                    inputs=[embedding_img, embedding_loc],
                    outputs=[embedding_box],
                    api_name=False
                )

                def visible_text_label(embedding_type, embedding_num):
                    if embedding_type == "input":
                        tip = "-".join([f"FFFF-{_}{_}{_}{_}" for _ in range(embedding_num)])
                        return gr.update(visible=True, label=f"Watermark Text (Format: {tip})")
                    else:
                        return gr.update(visible=False)

                def check_embedding_str(embedding_str, embedding_num):
                    if not re.match(r"^([0-9A-F]{4}-[0-9A-F]{4}-){%d}[0-9A-F]{4}-[0-9A-F]{4}$" % (embedding_num-1), embedding_str):
                        tip = "-".join([f"FFFF-{_}{_}{_}{_}" for _ in range(embedding_num)])
                        gr.Warning(f"Invalid format. Please use {tip}", duration=0)
                        return gr.update(interactive=False)
                    else:
                        return gr.update(interactive=True)

                embedding_num.change(
                    fn=visible_text_label,
                    inputs=[embedding_type, embedding_num],
                    outputs=[embedding_str],
                    api_name=False
                )
                embedding_type.change(
                    fn=visible_text_label,
                    inputs=[embedding_type, embedding_num],
                    outputs=[embedding_str],
                    api_name=False
                )
                embedding_str.change(
                    fn=check_embedding_str,
                    inputs=[embedding_str, embedding_num],
                    outputs=[embedding_btn],
                    api_name=False
                )

                embedding_btn.click(
                    fn=embed_watermark,
                    inputs=[embedding_img, embedding_num, embedding_type, embedding_str, embedding_loc],
                    outputs=[marked_image, marked_mask, marked_msg]
                )

            with gr.TabItem("Detect Watermark"):
                with gr.Row():
                    with gr.Column():
                        detecting_img = gr.Image(label="Input Image", type="numpy", height=512)
                    with gr.Column():
                        tip_md = gr.Markdown("""
                        **Note:** The split operation might not yield any results,
                        and subprocesses will be used to support timeout.

                        On the Windows platform, creating subprocesses will be noticeably slower.
                        """)
                        multi_ckb = gr.Checkbox(label="Split into multiple", value=False)
                        timeout_sli = gr.Slider(1, max_timeout, value=30, step=1, label="Timeout of multiple", visible=False)
                        detecting_btn = gr.Button("Detect Watermark")
                        predicted_messages = gr.JSON(label="Detected Messages")
                        color_cluster = gr.Markdown()
                with gr.Row():
                    predicted_mask = gr.Image(label="Predicted Watermark Position")
                    predicted_cluster = gr.Image(label="Watermark Clusters")

                detecting_img.change(
                    fn=lambda x: gr.update(value=False),
                    inputs=detecting_img,
                    outputs=multi_ckb,
                    api_name=False
                )
                multi_ckb.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=multi_ckb,
                    outputs=timeout_sli,
                    api_name=False
                )
                detecting_btn.click(
                    fn=detect_watermark,
                    inputs=[detecting_img, multi_ckb, timeout_sli],
                    outputs=[predicted_mask, predicted_cluster, predicted_messages, color_cluster]
                )

    if __name__ == '__main__':
        demo.launch(server_name="0.0.0.0", server_port=7860, mcp_server=True, ssr_mode=False)
