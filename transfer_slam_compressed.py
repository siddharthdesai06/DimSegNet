import clip
import os
import time
from typing import Literal
import torch
import tyro
from gsplat import rasterization
import pycolmap_scene_manager as pycolmap
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # To avoid conflict with cv2
from tqdm import tqdm
# from lseg import LSegNet
import json
import sys
import cv2
from typing import Literal
import tyro
from torchvision import transforms as T
from torchvision.models import resnet50
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

import torch.nn as nn




dev = "cuda"


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Parameter(torch.randn(512, 16))
        self.decoder = nn.Parameter(torch.randn(16, 512))

    def forward(self, x):
        x = x @ self.encoder
        y = x @ self.decoder
        return x, y


encoder_decoder = EncoderDecoder().to("cuda")
encoder_decoder.load_state_dict(torch.load("/home/siddharth/siddharth/thesis/my_seg_yolo/enc_dec_model/encoder_decoder.ckpt"))


class ObjectDetector:
    def __init__(self, model_name='detr_resnet50', conf_threshold=0.5, save_folder = "detected_images"):
        self.conf_threshold = conf_threshold
        self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
        self.model.eval()
        self.model.to(dev)
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)  # Create folder if it doesn't exist
        self.image_counter = len(os.listdir(self.save_folder))  # Track image count for naming

    
    def preprocess(self, image):
        return self.transform(image).unsqueeze(0)
    
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def plot_results(self, pil_img, prob, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{class_names[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    def detect_objects(self, image):
        img_tensor = self.preprocess(image).to(dev)
        outputs = self.model(img_tensor)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.conf_threshold
        # print(image.size)
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
        # self.plot_results(image, probas[keep], bboxes_scaled)
        predicted_class_indices = probas.argmax(-1)[keep]

        # self.draw_and_save_results(image, probas[keep], bboxes_scaled, image_name)
         # Save image with bounding boxes
        save_path = os.path.join(self.save_folder, f"image{self.image_counter}.jpg")
        self.image_counter += 1  # Increment counter
        
        self.draw_and_save_results(image, probas[keep], bboxes_scaled, save_path)
        print("saving.........")
        # sys.exit
        return bboxes_scaled, predicted_class_indices

    def draw_and_save_results(self, pil_img, prob, boxes, save_path):
        """Draws bounding boxes on an image and saves the output without displaying."""
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()  # Use a basic font (change if needed)
        print("save apth is", save_path)
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)  # Draw bounding box
            cl = p.argmax()  # Get class index
            text = f'{class_names[cl]}: {p[cl]:.2f}'  # Format label text
            draw.text((xmin, ymin), text, fill="yellow", font=font)  # Add label

        pil_img.save(save_path)  # Save the image



class SAMSegmenter:
    def __init__(self, sam_checkpoint, device=dev):
        self.model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.model.to(dev)
        self.predictor = SamPredictor(self.model)
    
    def set_image(self, image):
        self.predictor.set_image(image)
    
    def segment_objects(self, bboxes):
        all_masks = []
        for box in bboxes:
            masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
            all_masks.append(masks)
        return np.array(all_masks)
    
# class CLIPFeatureExtractor:
#     def __init__(self, embeddings_path):
#         self.embeddings = np.load(embeddings_path, allow_pickle=True).item()
    
#     def generate_feature_map(self, image, masks, class_indices, classes):
#         if masks.is_cuda:
#             masks = masks.cpu()
#         H, W, _ = np.array(image).shape
#         feature_map = np.zeros((H, W, 512), dtype=np.float32)
#         for i, mask in enumerate(masks):
#             class_idx = class_indices[i]
#             if class_idx >= len(classes):
#                 continue
#             class_vector = self.embeddings[classes[class_idx]]
#             feature_map[mask> 0] = class_vector
#         return feature_map



class CLIPFeatureExtractor:
    def __init__(self, embeddings_path):
        # Load the pre-trained embeddings from the given path
        self.embeddings = np.load(embeddings_path, allow_pickle=True).item()

        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Compute the "others" embedding once and store it for future use
        self.others_embedding = self._get_others_embedding()

    def _get_others_embedding(self):
        # Compute the "others" embedding only once
        prompt = ["others"]
        inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True).to("cuda")
        text_feat = self.clip_model.get_text_features(**inputs)  # Shape: [1, 512] (embedding for "others")
        others_embedding = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        
        # Detach the tensor and convert it to a NumPy array
        return others_embedding.detach().cpu().numpy()

    def generate_feature_map(self, image, masks, class_indices, classes):
        if masks.is_cuda:
            masks = masks.cpu()

        # Use the precomputed "others" embedding
        others_embedding = self.others_embedding

        # Get the shape of the image
        H, W, _ = np.array(image).shape
        
        # Initialize the feature map with the "others" embedding
        feature_map = others_embedding.repeat(H * W, axis=0).reshape(H, W, 512)

        # Apply the class-specific embeddings for each mask
        for i, mask in enumerate(masks):
            class_idx = class_indices[i]
            if class_idx >= len(classes) or classes[class_idx] == "N/A":
                continue
            class_vector = self.embeddings[classes[class_idx]]

            feature_map[mask > 0] = class_vector

        return feature_map

class Visualizer:
    @staticmethod
    def plot_segmentation(image, masks, save_dir="segmentations_replica", image_id=0):
        """Saves the segmentation visualization without displaying it."""
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
        save_path = os.path.join(save_dir, f"image{image_id}.png")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(image)

        for mask in masks:
            # Ensure the mask is a NumPy array
            mask = mask.cpu().numpy()  # Convert tensor to NumPy array (if it's on GPU, use .cpu())
            mask = mask.astype(np.uint8) * 255  # Convert to uint8 and scale to 255
            
            if np.any(mask > 0):  # Only process non-empty masks
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=np.random.rand(3), linewidth=2)

        ax.axis('off')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Ensure no display

    @staticmethod
    def show_binary_mask(mask, image_id=None):
        if len(mask) == 0:
            print("No masks found.")
            return
        
        # Move tensor to CPU if it's on GPU and convert to numpy array
        if mask.is_cuda:
            mask = mask.cpu().numpy()  # Move to CPU and convert to numpy

        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap="gray")  # Display binary mask in black & white
        plt.axis("off")
        
        # Optionally save the image (you can add a name pattern)
        if image_id is not None:
            save_path = f"mask_{image_id}.png"
            plt.savefig(save_path, bbox_inches="tight")
        
        plt.close()  # Close the plot after saving# Ensure no display


# # Configurable variables
IMAGE_PATH = "/home/siddharth/siddharth/thesis/3dgs-gradient-backprojection/data/garden/images/DSC07956.JPG"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
CLIP_EMBEDDINGS_PATH = "clip_coco_embeddings_hf.npy"
SAVE_DIR = "lang_feat"



# classes
class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush', 80: 'others'
}

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def torch_to_cv(tensor):
    img_cv = tensor.detach().cpu().numpy()[..., ::-1]
    img_cv = np.clip(img_cv * 255, 0, 255).astype(np.uint8)
    return img_cv


def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d


def to_tensor_safe(data):
    if isinstance(data, torch.Tensor):
        return data.clone().detach().float()
    return torch.tensor(data).float()

def splats_to_tensor(splats):
    splats["means"] = to_tensor_safe(splats["means"])
    splats["rotation"] = to_tensor_safe(splats["rotation"])
    features_dc = to_tensor_safe(splats["features_dc"])
    features_rest = to_tensor_safe(splats["features_rest"])
    splats["opacity"] = to_tensor_safe(splats["opacity"])
    splats["scaling"] = to_tensor_safe(splats["scaling"])
    splats["features_dc"] = features_dc[:,None,:].clone()
    splats["features_rest"] = features_rest.reshape(-1, 15, 3).clone()
    
    return splats
    
def load_checkpoint(
    checkpoint: str,
    camera_json: str,
    rasterizer: Literal["inria", "gsplat"] = "inria",
    data_factor: int = 1,
):
    
    assert os.path.exists(camera_json), f"Camera JSON file {camera_json} does not exist."
    with open(camera_json, "r") as f:
        cameras = json.load(f)

    model = torch.load(checkpoint)  # Make sure it is generated by 3DGS original repo
    # sys.exit()
    if rasterizer == "inria":
        print("using inria rasterizer")
        model_params = model
        # print(model_params.keys())
        splats = {  
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["features_dc"],
            "features_rest": model_params["features_rest"],
            "scaling": model_params["scaling"],
            "rotation": model_params["rotation"],
            "opacity": model_params["opacity"].squeeze(1),
        }
    elif rasterizer == "gsplat":
        print("using gsplat rasterizer")
        print(model["splats"].keys())
        model_params = model["splats"]
        splats = {
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["sh0"],
            "features_rest": model_params["shN"],
            "scaling": model_params["scales"],
            "rotation": model_params["quats"],
            "opacity": model_params["opacities"],
        }
    else:
        raise ValueError("Invalid rasterizer")

    _detach_tensors_from_dict(splats)

    first_cam = cameras[0]
    fx,fy = first_cam["fx"], first_cam["fy"]
    cx,cy = first_cam["width"]/2, first_cam["height"]/2

    # Assuming only one camera
    camera_matrix = torch.tensor(
        [
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1],
        ]
    )
    camera_matrix[:2, :3] /= data_factor
    
    splats["camera_matrix"] = camera_matrix
    splats["slam_positions"] = [
        {"id":cam["id"], "position": cam["position"],"rotation": cam["rotation"]} 
        for cam in cameras
    ]
    splats["camera_json"] = camera_json
    splats = splats_to_tensor(splats)
    
    return splats

def get_viewmat_position_and_rotation(position,rotation):
    # Convert inputs to tensors
    position = torch.tensor(position, dtype=torch.float)
    rotation = torch.tensor(rotation, dtype=torch.float)
    
    # Construct the view matrix
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = torch.tensor(rotation)
    viewmat[:3, 3] = torch.tensor(position)

    viewmat = torch.inverse(viewmat)
    
    return viewmat

def prune_by_gradients(splats):
    
    means = splats["means"]
    quats = splats["rotation"]
    features_dc = splats["features_dc"]
    features_rest = splats["features_rest"]
    opacities = splats["opacity"]
    scales = splats["scaling"]

    # print("means",means.shape)
    # print("quats",quats.shape)
    # print("features_dc",features_dc.shape)
    # print("features_rest",features_rest.shape)
    # print("opacities",opacities.shape)
    # print("scales",scales.shape)
    
    colors = torch.cat([features_dc, features_rest], dim=1)
    
    # print("features_dc_later",features_dc.shape)
    # print("features_rest_later",features_rest.shape)
    # print("colors_later",colors.shape)

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    
    K = splats["camera_matrix"]
    slam_positions = splats["slam_positions"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)
    
    for cam in slam_positions:
        viewmat = get_viewmat_position_and_rotation(cam["position"], cam["rotation"])
        
        output,_,_=rasterization(
            means,
            quats,
            scales,
            opacities,
            colors[:,0,:],
            viewmats = viewmat[None],
            Ks=K[None],
            width = K[0,2]*2,
            height=K[1,2]*2,
            backgrounds=torch.ones((1,3)).to(colors.device),
        )

        output_cv = torch_to_cv(output[0])
        cv2.imshow("output", output_cv)
        cv2.waitKey(100)
        
        #Compute pseudo loss and backpropagate
        pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
        pseudo_loss.backward()
        gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
        colors.grad.zero_()
    
    mask = gaussian_grads > 0
    print("Total splats", len(gaussian_grads))
    print("Pruned", (~mask).sum().item(), "splats")
    print("Remaining", mask.sum().item(), "splats")
    
    
    # Apply the mask to prune the splats
    pruned_splats = splats.copy()
    pruned_splats["means"] = means[mask]
    pruned_splats["features_dc"] = features_dc[mask]
    pruned_splats["features_rest"] = features_rest[mask]
    pruned_splats["scaling"] = scales[mask]
    pruned_splats["rotation"] = quats[mask]
    pruned_splats["opacity"] = opacities[mask]
    
    return pruned_splats

def test_proper_pruning(splats, splats_after_pruning):
    # colmap_project = splats["colmap_project"]
    # frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]

    means_pruned = splats_after_pruning["means"]
    colors_dc_pruned = splats_after_pruning["features_dc"]
    colors_rest_pruned = splats_after_pruning["features_rest"]
    colors_pruned = torch.cat([colors_dc_pruned, colors_rest_pruned], dim=1)
    opacities_pruned = torch.sigmoid(splats_after_pruning["opacity"])
    scales_pruned = torch.exp(splats_after_pruning["scaling"])
    quats_pruned = splats_after_pruning["rotation"]

    K = splats["camera_matrix"]
    slam_positions = splats["slam_positions"]
    total_error = 0
    max_pixel_error = 0
    for cam in slam_positions:
        viewmat = get_viewmat_position_and_rotation(cam["position"], cam["rotation"])
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        output_pruned, _, _ = rasterization(
            means_pruned,
            quats_pruned,
            scales_pruned,
            opacities_pruned,
            colors_pruned,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        total_error += torch.abs((output - output_pruned)).sum()
        max_pixel_error = max(
            max_pixel_error, torch.abs((output - output_pruned)).max()
        )

    percentage_pruned = (
        (len(splats["means"]) - len(splats_after_pruning["means"]))
        / len(splats["means"])
        * 100
    )

    # assert max_pixel_error < 1 / (
    #     255 * 2
    # ), "Max pixel error should be less than 1/(255*2), safety margin"
    print(
        "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
            percentage_pruned, max_pixel_error, total_error
        )
    )


def create_feature_field_detr_sam_clip(splats, sam_checkpoint, clip_embeddings_path, batch_count=1, use_cpu=False):
    device = "cpu" if use_cpu else "cuda"

    detector = ObjectDetector()
    segmenter = SAMSegmenter(sam_checkpoint)
    clip_extractor = CLIPFeatureExtractor(clip_embeddings_path)

    means = splats["means"].to(device)
    colors_dc = splats["features_dc"].to(device)
    colors_rest = splats["features_rest"].to(device)
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)
    
    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    opacities = torch.sigmoid(splats["opacity"]).to(device)
    scales = torch.exp(splats["scaling"]).to(device)
    quats = splats["rotation"].to(device)
    K = splats["camera_matrix"].to(device)
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors_dc.shape[0], 512, device=device)
    gaussian_denoms = torch.ones(colors_dc.shape[0], device=device) * 1e-12
    
    colors_feats = torch.zeros(colors.shape[0], 512, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    slam_positions = splats["slam_positions"]
    # images = sorted(colmap_project.images.values(), key=lambda x: x.name)
    # batch_size = max(1, len(images) // batch_count) if batch_count > 0 else 1
    image_id = 0
    for cam in tqdm(splats["slam_positions"], desc="Feature backprojection"):
        # batch = images[batch_start:batch_start + batch_size]
        # for image in batch:
            viewmat = get_viewmat_position_and_rotation(cam["position"], cam["rotation"])
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)
            
            with torch.no_grad():
                output, _, meta = rasterization(
                    means, 
                    quats, 
                    scales, 
                    opacities, 
                    colors_all, 
                    viewmat[None], 
                    K[None], 
                    width, 
                    height, 
                    sh_degree=3,
                    )
                image_tensor = output.permute(0, 3, 1, 2).to(device)
                
                # Convert rasterized output to PIL image
                image_np = output[0].cpu().numpy()  
                image = Image.fromarray((image_np * 255).astype(np.uint8)) 

                # Run DETR to get bounding boxes and class predictions
                bboxes, class_indices = detector.detect_objects(image)
                
                print("keep", class_indices)
                # labels = [CLASSES[i] for i in class_indices]
                # print(labels)
                
                # Use SAM to get masks
                segmenter.set_image(image_np)
                masks = segmenter.segment_objects(bboxes.detach().cpu().numpy())
                print(masks.shape)
                # sys.exit()
                # Visualizer.plot_segmentation(image_np, masks.squeeze(1))
                # Visualizer.show_binary_mask(masks[0].squeeze(0))
             
                if isinstance(masks, np.ndarray):
                    masks = torch.tensor(masks)  # Convert to PyTorch tensor

                # Get CLIP feature map
                if masks.numel() > 0:  # Ensure masks are not empt 
                    Visualizer.plot_segmentation(image_np, masks.squeeze(1), image_id=image_id)
                    # Visualizer.show_binary_mask(masks[0].squeeze(0), image_id=image_id)
                    clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, CLASSES)
   
                # clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, CLASSES)
                feats = torch.nn.functional.normalize(torch.tensor(clip_feature_map, device = dev), dim=-1)

                image_id+=1

            # Backproject features onto gaussians
            output_for_grad, _, meta = rasterization(
                means, 
                quats, 
                scales, 
                opacities, 
                colors_feats, 
                viewmat[None], 
                K[None], 
                width, 
                height)
            
            target = (output_for_grad[0].to(device) * feats).sum()
            target.backward()
            colors_feats_copy = colors_feats.grad.clone()
            colors_feats.grad.zero_()

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target_0 = (output_for_grad[0]).sum()
            target_0.to(device)
            
            target_0.backward()

            gaussian_features += colors_feats_copy
            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()

            del viewmat, meta, _, output, feats, output_for_grad, colors_feats_copy, target, target_0
            torch.cuda.empty_cache()
            
            # Normalize and handle NaNs
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    
    gaussian_features[torch.isnan(gaussian_features)] = 0
    return gaussian_features



def create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embeddings_path, embed_dim=512, batch_count=1, use_cpu=False):
    device = "cpu" if use_cpu else "cuda"

    # detector = ObjectDetector()
    yolo_model = YOLO('yolov12x.pt')
    segmenter = SAMSegmenter(sam_checkpoint)
    clip_extractor = CLIPFeatureExtractor(clip_embeddings_path)

    means = splats["means"].to(device)
    colors_dc = splats["features_dc"].to(device)
    colors_rest = splats["features_rest"].to(device)
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)
    
    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    opacities = torch.sigmoid(splats["opacity"]).to(device)
    scales = torch.exp(splats["scaling"]).to(device)
    quats = splats["rotation"].to(device)
    K = splats["camera_matrix"].to(device)
    colors.requires_grad = True
    colors_0.requires_grad = True

    # embed_dim = 
    gaussian_features = torch.zeros(colors_dc.shape[0], embed_dim, device=device)
    gaussian_denoms = torch.ones(colors_dc.shape[0], device=device) * 1e-12
    
    colors_feats = torch.zeros(colors.shape[0], embed_dim, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    slam_positions = splats["slam_positions"]
    # images = sorted(colmap_project.images.values(), key=lambda x: x.name)
    # batch_size = max(1, len(images) // batch_count) if batch_count > 0 else 1
    image_id = 0
    for cam in tqdm(splats["slam_positions"], desc="Feature backprojection"):
        # batch = images[batch_start:batch_start + batch_size]
        # for image in batch:
            viewmat = get_viewmat_position_and_rotation(cam["position"], cam["rotation"])
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)
            
            with torch.no_grad():
                output, _, meta = rasterization(
                    means, 
                    quats, 
                    scales, 
                    opacities, 
                    colors_all, 
                    viewmat[None], 
                    K[None], 
                    width, 
                    height, 
                    sh_degree=3,
                    )
                image_tensor = output.permute(0, 3, 1, 2).to(device)
                
                # Convert rasterized output to PIL image
                image_np = output[0].cpu().numpy()  
                image = Image.fromarray((image_np * 255).astype(np.uint8)) 

                # Run yolo to get bounding boxes and class predictions
                results = yolo_model(image)
                detections = results[0].boxes
                class_indices = detections.cls.int().tolist()
                bboxes =detections.xyxy.cpu().numpy()
                print("YOLO Detected Classes:", class_indices)

                # print("keep", class_indices)
                # labels = [CLASSES[i] for i in class_indices]
                # print(labels)
                
                # Use SAM to get masks
                segmenter.set_image(image_np)
                masks = segmenter.segment_objects(bboxes)
                print(masks.shape)
                # sys.exit()
                # Visualizer.plot_segmentation(image_np, masks.squeeze(1))
                # Visualizer.show_binary_mask(masks[0].squeeze(0))

                
                
                if isinstance(masks, np.ndarray):
                    masks = torch.tensor(masks)  # Convert to PyTorch tensor

                # Get CLIP feature map
                if masks.numel() > 0:  # Ensure masks are not empt 
                    Visualizer.plot_segmentation(image_np, masks.squeeze(1), image_id=image_id)
                    # Visualizer.show_binary_mask(masks[0].squeeze(0), image_id=image_id)
                    clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, class_names)
   
                # clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, CLASSES)
                feats = torch.nn.functional.normalize(torch.tensor(clip_feature_map, device = dev), dim=-1)
                print("feats shape before",feats.shape)
                feats = feats @ encoder_decoder.encoder #(512->16)
                print("feats shape after",feats.shape)
                image_id+=1

            # Backproject features onto gaussians
            output_for_grad, _, meta = rasterization(
                means, 
                quats, 
                scales, 
                opacities, 
                colors_feats, 
                viewmat[None], 
                K[None], 
                width, 
                height)
            
            target = (output_for_grad[0].to(device) * feats).sum()
            target.backward()
            colors_feats_copy = colors_feats.grad.clone()
            colors_feats.grad.zero_()

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target_0 = (output_for_grad[0]).sum()
            target_0.to(device)
            
            target_0.backward()

            gaussian_features += colors_feats_copy
            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()

            del viewmat, meta, _, output, feats, output_for_grad, colors_feats_copy, target, target_0
            torch.cuda.empty_cache()
            
            # Normalize and handle NaNs
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    
    gaussian_features[torch.isnan(gaussian_features)] = 0
    return gaussian_features




def main(
    json_directory: str = "/home/siddharth/siddharth/thesis/RTG-SLAM/output/dataset/Replica/office0/cameras.json",  # camera json file
    checkpoint: str = "/home/siddharth/siddharth/thesis/RTG-SLAM/output/dataset/Replica/office0/save_model/frame_2000/iter_1139_stable.pth",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "/home/siddharth/siddharth/thesis/my_seg_yolo/output/replica/office0",  # output path
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    clip_embedding_path: str = "clip_coco_embeddings_hf.npy",
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "inria",  # Original or GSplat for checkpoints
    data_factor: int = 4,
    embed_dim: int=16, # the dimension to which you trained enc-dec
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, json_directory, rasterizer=rasterizer, data_factor=data_factor
    )
    # splats_optimized = prune_by_gradients(splats)
    
    # # print("Pruning done now exiting")
    # test_proper_pruning(splats, splats_optimized)
    # print("Proper Pruning done now exiting")
    # sys.exit()
    
    # splats = splats_optimized
    # features = create_feature_field_lseg(splats)
    features = create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embedding_path,embed_dim)

    print(features.shape)
    torch.save(features, f"{results_dir}/features_detr.pt")

if __name__ == "__main__":
    tyro.cli(main)
