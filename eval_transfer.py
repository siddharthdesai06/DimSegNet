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
import random
from matplotlib import patches
from sklearn.decomposition import PCA
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
        with torch.no_grad():
            text_feat = self.clip_model.get_text_features(**inputs)  # Shape: [1, 512] (embedding for "others")
            others_embedding = torch.nn.functional.normalize(text_feat, dim=-1)
        
        # Detach the tensor and convert it to a NumPy array
        return others_embedding.detach().cpu().numpy()

    def generate_feature_map(self, image, masks, class_indices, classes):
        if masks.is_cuda:
            masks = masks.cpu()

        # Use the precomputed "others" embedding
        others_embedding = self.embeddings["others"]  
        others_embedding = others_embedding.reshape(1, -1)   

        # other_calc= self.others_embedding

        # others_embedding = torch.tensor(others_embedding) if isinstance(others_embedding, np.ndarray) else others_embedding
        # other_calc = torch.tensor(other_calc) if isinstance(other_calc, np.ndarray) else other_calc
        # others_embedding = torch.nn.functional.normalize(others_embedding,dim=-1)


        # Get the shape of the image
        H, W, _ = np.array(image).shape

        # print("others embed shape",others_embedding.shape)
        # print("others embed shape",other_calc.shape)

        # Initialize the feature map with the "others" embedding
        feature_map = others_embedding.repeat(H * W, axis=0).reshape(H, W, 512)
        np.set_printoptions(precision=10, suppress=False)
        # print("Feature Map:", feature_map[0][0])
        # sys.exit()
        # print(feature_map.shape)
        # print(feature_map[0][0])
        # print("class_indices",len(class_indices))
        # print("masks_shape",masks.shape)
        # sys.exit()
        # Apply the class-specific embeddings for each mask
        for i, mask in enumerate(masks):
            class_idx = class_indices[i]
            if class_idx >= len(classes) or classes[class_idx] == "N/A":
                continue
            # print("embeddeong attachment",classes[class_idx])
            # sys.exit()
            class_vector = self.embeddings[classes[class_idx]]
            # print(class_vector)
            # sys.exit()
            # print("mask ka shape",mask.shape)
            feature_map[mask > 0] = class_vector
            # print("class_vector for that class",class_vector)
            # indices = np.argwhere(mask > 0)  # Get [x, y] coordinates
            # indices = indices.T
            # print("indices ak shape",indices.shape)
            # for x, y in indices:
            #     print(f"Feature at ({x}, {y}): {feature_map[x, y]}")
            #     break
        return feature_map

class Visualizer:

    @staticmethod
    def save_yolo_op(image, bboxes, masks, class_names, save_dir="visualizations/yolo_output", image_id=0):
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
        save_path = os.path.join(save_dir, f"image_{image_id}.png")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)    

        # Draw bounding boxes for YOLO detections
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(bboxes))]
        for i, detection in enumerate(bboxes):
            x1, y1, x2, y2 = detection
            class_name = class_names[i]
            color = tuple([c / 255.0 for c in colors[i]])

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.text(x1, y1 - 10, class_name, color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Ensure no display

        # print(f"Saved visualization to {save_path}")

    
    def save_sam_op(masks, save_dir="visualizations/sam_output", image_id=0):
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists

        for i, mask in enumerate(masks):
            # Convert mask to binary format (255 for object, 0 for background)
            binary_mask = (mask[0].cpu().numpy() * 255).astype(np.uint8)  

            # Save the binary mask as an image
            mask_filename = os.path.join(save_dir, f"mask_{image_id}_{i}.png")
            cv2.imwrite(mask_filename, binary_mask)

            # print(f"Saved binary mask: {mask_filename}")

    def save_overlayed_segmentation(image, masks, save_dir="visualizations/segmentations", image_id=0):
        os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
        save_path = os.path.join(save_dir, f"image{image_id}.png")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(image)
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = mask.astype(np.uint8) * 255  
            
            if np.any(mask > 0): 
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=np.random.rand(3), linewidth=2)

        ax.axis('off')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  

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

def prune_by_gradients(splats):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors[:, 0, :],
            viewmats=viewmat[None],
            Ks=K[None],
            # sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )
        frame_idx += 1
        pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
        pseudo_loss.backward()
        # print(colors.grad.shape)
        gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
        colors.grad.zero_()

    mask = gaussian_grads > 0
    print("Total splats", len(gaussian_grads))
    print("Pruned", (~mask).sum(), "splats")
    print("Remaining", mask.sum(), "splats")
    splats = splats.copy()
    splats["means"] = splats["means"][mask]
    splats["features_dc"] = splats["features_dc"][mask]
    splats["features_rest"] = splats["features_rest"][mask]
    splats["scaling"] = splats["scaling"][mask]
    splats["rotation"] = splats["rotation"][mask]
    splats["opacity"] = splats["opacity"][mask]
    return splats


def test_proper_pruning(splats, splats_after_pruning):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
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
    total_error = 0
    max_pixel_error = 0
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
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

    assert max_pixel_error < 1 / (
        255 * 2
    ), "Max pixel error should be less than 1/(255*2), safety margin"
    print(
        "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
            percentage_pruned, max_pixel_error, total_error
        )
    )


def create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embeddings_path, embed_dim=512, compress = False, test_images={},batch_count=1, use_cpu=False):
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

    gaussian_features = torch.zeros(colors.shape[0], embed_dim, device=device)
    gaussian_denoms = torch.ones(colors.shape[0], device=device) * 1e-12
    
    colors_feats = torch.zeros(colors.shape[0], embed_dim, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    colmap_project = splats["colmap_project"]
    images = sorted(colmap_project.images.values(), key=lambda x: x.name)
    image_id = 0

    pca = PCA(n_components=3)
    first_time = True
    unique_labels = set()
    for image in tqdm(images, desc="Feature backprojection (images)"):
            if image.name in test_images:
                print(f"Skipping {image.name} as it is test image")
                continue
            viewmat = get_viewmat_from_colmap_image(image)
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)
            with torch.no_grad():
                output, alphas, meta = rasterization(
                    means, 
                    quats, 
                    scales, 
                    opacities, 
                    colors_all, 
                    viewmat[None], 
                    K[None], 
                    width=width, 
                    height=height, 
                    sh_degree=3,
                    )
                # image_tensor = output.permute(0, 3, 1, 2).to(device)

                alpha = alphas[0].detach().cpu().numpy()
                cv2.imshow("Alpha", alpha)
                
                # Convert rasterized output to PIL image
                image_np = output[0].cpu().numpy()  
                image = Image.fromarray((image_np * 255).astype(np.uint8)) 

                results = yolo_model(image, verbose=False)

                detections = results[0].boxes
                class_indices = detections.cls.int().tolist()
                bboxes =detections.xyxy.cpu().numpy()

                # print("YOLO Detected Classes:", class_indices)
                labels = [class_names[i] for i in class_indices]
                unique_labels.update(labels)
                print(labels)

                # Use SAM to get masks
                segmenter.set_image(image_np)
                masks = segmenter.segment_objects(bboxes)

                if isinstance(masks, np.ndarray):
                    masks = torch.tensor(masks)  # Convert to PyTorch tensor

                # Get CLIP feature map
                flag = True
                if masks.numel() > 0:  # Ensure masks are not empt 
                    
                    Visualizer.save_yolo_op(image, bboxes, masks, labels, image_id=image_id)
                    Visualizer.save_overlayed_segmentation(image_np, masks.squeeze(1), image_id=image_id)
                    Visualizer.save_sam_op(masks=masks, image_id=image_id)

                    clip_feature_map = clip_extractor.generate_feature_map(image, masks.squeeze(1), class_indices, class_names)
                    flag = False

                    mask_2d = masks.squeeze(1).cpu().numpy()
                    for mask in mask_2d:
                        image_np = image_np * (1-0.5*mask[...,None]) + (mask[...,None])*np.array([0, 0.0, 1.0])
                        # print(image_np.shape)
                        image_np = np.clip(image_np, 0, 1.0)
                        image_np = image_np.astype(np.float32)
                    # exit()
                    image_np = np.ascontiguousarray(image_np)
                    cv2.imshow("Image with mask", image_np)
                    

#---------------------------------------------------------------------------
                # print("clip_feature_map_shape",clip_feature_map.shape)
                # print("before",clip_feature_map[18,34])
                feats = torch.nn.functional.normalize(torch.tensor(clip_feature_map, device = dev), dim=-1)
                

                # print("after",feats[18,34])
                # sys.exit()
#----------------------------------------------------------------

                print("feats_feature_map_shape",feats.shape)
                # sys.exit()
                # for 512->16 
                if compress:
                    feats = feats @ encoder_decoder.encoder 
        
                image_id+=1

            image_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  
            print("Final Feats shape", feats.shape)
            feats_flattened = feats.reshape(-1, 512).detach().cpu().numpy()
            if first_time:
                feats_pca = pca.fit_transform(feats_flattened)
                feats_pca_first_time = feats_pca
                first_time = False
            else:
                feats_pca = pca.transform(feats_flattened)
            feats_pca = (feats_pca - feats_pca_first_time.min(axis=0))/(feats_pca_first_time.max(axis=0) - feats_pca_first_time.min(axis=0))
            feats_pca = np.clip(feats_pca, 0, 1)
            feats_pca = feats_pca.reshape(height, width, 3)
            
            
            combined_image = np.hstack((image_original,(feats_pca * 255).astype(np.uint8)))
           
            # cv2.imshow("feats_pca", (feats_pca * 255).astype(np.uint8))
            cv2.imshow("combined+pca",combined_image)

            cv2.waitKey(1)

            if flag:
                continue

            # Backproject features onto gaussians
            output_for_grad, _, meta = rasterization(
                means, 
                quats, 
                scales, 
                opacities, 
                colors_feats, 
                viewmat[None], 
                K[None], 
                width=width, 
                height=height)
            
            target = (output_for_grad[0].to(device) * feats).sum()
            target.to(device)
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
            # break
            
            # Normalize and handle NaNs
    print(unique_labels)
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features[torch.isnan(gaussian_features)] = 0
    pca = PCA(n_components=3)
    print("gaussian_features.shape", gaussian_features.shape)
    gaussians_pca = pca.fit_transform(gaussian_features.detach().cpu().numpy())
    gaussians_pca = gaussians_pca - np.min(gaussians_pca, axis=1,keepdims=True)
    gaussians_pca = gaussians_pca / (np.max(gaussians_pca, axis=1,keepdims=True) + 1e-9)
    print("gaussians_pca.shape", gaussians_pca.shape)
    gaussians_pca = torch.from_numpy(gaussians_pca).float().cuda()
    for image in tqdm(images, desc="PCA of gaussians"):
            if image.name in test_images:
                print(f"Skipping {image.name} as it is test image")
                continue
            viewmat = get_viewmat_from_colmap_image(image)
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)
            output, _, meta = rasterization(
                    means, 
                    quats, 
                    scales, 
                    opacities, 
                    gaussians_pca, 
                    viewmat[None], 
                    K[None], 
                    width=width, 
                    height=height, 
                    # sh_degree=3,
                    )
            output_cv = torch_to_cv(output[0])
            # output_cv = np.clip(output_cv, 0, 1)
            cv2.imshow("pca", output_cv)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    return gaussian_features

def main(
    
    data_dir: str = "/home/siddharth/siddharth/thesis/Yolo_segmentation/eval_datasets/figurines",  # colmap path
    checkpoint: str = "/home/siddharth/siddharth/thesis/Yolo_segmentation/eval_datasets/figurines/chkpnt30000.pth",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/figurines",

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
    test_images = {"test_0.jpg", "test_1.jpg", "test_2.jpg", "test_3.jpg", "frame_00131.jpg"} 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )

    splats_optimized = prune_by_gradients(splats)
    print("Prunign done")
    test_proper_pruning(splats, splats_optimized)
    splats = splats_optimized
    # features = create_feature_field_detr_sam_clip(splats, sam_checkpoint, clip_embedding_path)

    features = create_feature_field_yolo_sam_clip(splats, sam_checkpoint, clip_embedding_path,embed_dim,compress, test_images)

    print("features_size", features.shape)
    
    torch.save(features, f"{results_dir}/features.pt")

if __name__ == "__main__":
    tyro.cli(main)