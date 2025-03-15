import torch
import numpy as np
from tqdm import tqdm
import cv2
import math
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.models import resnet50
from segment_anything import sam_model_registry, SamPredictor
import time
from gsplat import rasterization
import pycolmap_scene_manager as pycolmap
import sys
from transformers import CLIPProcessor, CLIPModel
from typing import Literal
import tyro
from ultralytics import YOLO
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
encoder_decoder.load_state_dict(torch.load("./encoder_decoder.ckpt"))

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
            masks, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
            all_masks.append(masks)
        return np.array(all_masks)

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
    def plot_segmentation(image, masks, save_dir="segmentations", image_id=0):
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
        save_path = os.path.join(save_dir, f"image{image_id}.png")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(image)

        for mask in masks:
            # Ensure the mask is a NumPy array
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            # mask = mask.cpu().numpy()  # Convert tensor to NumPy array (if it's on GPU, use .cpu())
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
        if isinstance(mask, torch.Tensor):
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
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
CLIP_EMBEDDINGS_PATH = "clip_coco_embeddings_hf.npy"
SAVE_DIR = "lang_feat"

# COCO classes
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

def get_viewmat_from_colmap_image(image):
    viewmat = torch.eye(4).float()  # .to(device)
    viewmat[:3, :3] = torch.tensor(image.R()).float()  # .to(device)
    viewmat[:3, 3] = torch.tensor(image.t).float()  # .to(device)
    return viewmat

def load_checkpoint(
    checkpoint: str,
    data_dir: str,
    rasterizer: Literal["inria", "gsplat"] = "inria",
    data_factor: int = 1,
):

    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    colmap_project = pycolmap.SceneManager(colmap_dir)
    colmap_project.load_cameras()
    colmap_project.load_images()
    
    model = torch.load(checkpoint, weights_only = False)  # Make sure it is generated by 3DGS original repo
    if rasterizer == "inria":
        print("using inria rasterizer")
        model_params, _ = model
        splats = {
            "active_sh_degree": model_params[0],
            "means": model_params[1],
            "features_dc": model_params[2],
            "features_rest": model_params[3],
            "scaling": model_params[4],
            "rotation": model_params[5],
            "opacity": model_params[6].squeeze(1),
        }
    elif rasterizer == "gsplat":
        print("using gsplat rasterizer")
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

    # Assuming only one camera
    for camera in colmap_project.cameras.values():
        camera_matrix = torch.tensor(
            [
                [camera.fx, 0, camera.cx],
                [0, camera.fy, camera.cy],
                [0, 0, 1],
            ]
        )
        break

    camera_matrix[:2, :3] /= data_factor

    # print(camera_matrix)
    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir
    # print("splats_colmap_pro",splats["colmap_project"])
    # sys.exit()
    return splats

def create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embeddings_path, embed_dim=512, compress = False, batch_count=1, use_cpu=False):
    device = "cpu" if use_cpu else "cuda"

    if compress:
        print(f"Compressing the feature dimension to {embed_dim}")
        embed_dim=embed_dim
    else:
        embed_dim=512
        print(f"Not compressing, dimension kept is {embed_dim}")
        
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

    gaussian_features = torch.zeros(colors_dc.shape[0], embed_dim, device=device)
    gaussian_denoms = torch.ones(colors_dc.shape[0], device=device) * 1e-12
    
    colors_feats = torch.zeros(colors.shape[0], embed_dim, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    colmap_project = splats["colmap_project"]
    images = sorted(colmap_project.images.values(), key=lambda x: x.name)
    image_id = 0
    for image in tqdm(images, desc="Feature backprojection (images)"):
            viewmat = get_viewmat_from_colmap_image(image)
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

                results = yolo_model(image, verbose=False)
                detections = results[0].boxes
                class_indices = detections.cls.int().tolist()
                bboxes =detections.xyxy.cpu().numpy()

                print("YOLO Detected Classes:", class_indices)
                labels = [class_names[i] for i in class_indices]
                print(labels)
                # Use SAM to get masks
                segmenter.set_image(image_np)
                masks = segmenter.segment_objects(bboxes)

                # print(masks.shape)
                # # sys.exit()
                Visualizer.plot_segmentation(image_np, masks.squeeze(1))
                Visualizer.show_binary_mask(masks[0].squeeze(0))

                if isinstance(masks, np.ndarray):
                    masks = torch.tensor(masks)  # Convert to PyTorch tensor

                # Get CLIP feature map
                if masks.numel() > 0:  # Ensure masks are not empt 
                    Visualizer.plot_segmentation(image_np, masks.squeeze(1), image_id=image_id)
                    clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, class_names)
                feats = torch.nn.functional.normalize(torch.tensor(clip_feature_map, device = dev), dim=-1)
                
                # for 512->16 
                if compress:
                    feats = feats @ encoder_decoder.encoder 
        
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
    data_dir: str = "/home/siddharth/siddharth/thesis/Yolo_segmentation/eval_datasets/teatime",  # colmap path
    checkpoint: str = "/home/siddharth/siddharth/thesis/Yolo_segmentation/eval_datasets/teatime/chkpnt30000.pth",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/teatime",

    # data_dir: str = "/home/siddharth/siddharth/thesis/3dgs-gradient-backprojection/data/garden",  # colmap path
    # checkpoint: str = "/home/siddharth/siddharth/thesis/3dgs-gradient-backprojection/data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    # results_dir: str = "./results/garden",

    sam_checkpoint: str = "./sam_vit_h_4b8939.pth",
    clip_embedding_path: str = "clip_coco_embeddings_hf.npy",
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "inria",  # Original or GSplat for checkpoints
    data_factor: int = 4,
    embed_dim: int=16,
    compress: bool=False,
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )

    # splats_optimized = prune_by_gradients(splats)
    # print("Prunign done")
    # test_proper_pruning(splats, splats_optimized)
    # splats = splats_optimized
    # features = create_feature_field_detr_sam_clip(splats, sam_checkpoint, clip_embedding_path)

    features = create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embedding_path,embed_dim,compress)

    print("features_size", features.shape)
    
    torch.save(features, f"{results_dir}/features.pt")

if __name__ == "__main__":
    tyro.cli(main)