import torch
import clip
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import torch.nn as nn
import time
import os
import torch.optim as optim

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The device used is {device}")

# Load the pretrained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class_names_list = list(class_names.values())

# Compute CLIP embeddings for COCO classes
def compute_clip_class_embeddings(classes, clip_model, clip_processor):
    inputs = clip_processor(text=classes, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)  # (num_classes, 512)
        text_features = torch.nn.functional.normalize(text_features,dim=-1)  # Normalize

    # Store in a dictionary
    class_embeddings = {cls: text_features[i].cpu().numpy() for i, cls in enumerate(classes)}
    return class_embeddings

# Get and save embeddings
clip_class_embeddings = compute_clip_class_embeddings(class_names_list, clip_model, clip_processor)
np.save("clip_coco_embeddings_hf.npy", clip_class_embeddings)

embedding_matrix = torch.stack([torch.tensor(clip_class_embeddings[cls]) for cls in class_names_list])
embedding_matrix = embedding_matrix.to(device)

print(f"Embedding matrix created of shape {embedding_matrix.shape}")

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Parameter(torch.randn(512, 64))
        self.decoder = nn.Parameter(torch.randn(64, 512))

    def forward(self, x):
        x = x @ self.encoder
        y = x @ self.decoder
        return x, y
    

def train(model, embedding_matrix, device, num_epochs, save_path, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x_encoded, x_reconstructed = model(embedding_matrix)
        loss = criterion(x_reconstructed, embedding_matrix)  # Reconstruction loss
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training Complete!")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

model = EncoderDecoder()

num_epochs = 100000
save_path = "./encoder_decoder.ckpt"

train(model, embedding_matrix, device, num_epochs, save_path)
