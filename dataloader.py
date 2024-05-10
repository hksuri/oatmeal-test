import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
    
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.json_file = os.path.join(self.root_dir, '_annotations.coco.json')
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.images = self.data['images']
        self.annotations = {ann['id']: ann for ann in self.data['annotations']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations
        image_id = image_info['id']
        ann = self.annotations.get(image_id, {})
        masks = []
        boxes = []
        labels = []

        # Create masks
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        polygon = np.array(ann['segmentation'][0]).reshape((-1, 2)).astype(np.int32) 
        cv2.fillPoly(mask, [polygon], 1)
        masks.append(mask)
        
        # Bounding boxes and labels
        bbox = ann['bbox']
        boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        labels.append(ann['category_id'])
        
        # Torch Tensors
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.transform:
            image = self.transform(image)

        return image, target

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_dataloader(data_path, batch_size):
    dataset = BrainTumorDataset(data_path, transform=get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader
