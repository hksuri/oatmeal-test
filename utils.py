import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = transform_targets(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    return running_loss / len(data_loader)

def evaluate(model, data_loader, device, epoch, num_epochs, output_dir, num_images=5):
    model.eval()
    val_loss = 0
    image_counter = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = transform_targets(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images, targets)

            # loss_dict = model(images, targets)                    # Commented out because I am not getting val loss directly in
            # losses = sum(loss for loss in loss_dict.values())     # eval mode, went to MaskRCNN repo to check for issues, this
            # val_loss += losses.item()                             # required modifying original MaskRCNN so I am leaving it for now
            
            if epoch == num_epochs - 1 and image_counter < num_images:
                for i in range(len(images)):
                    if image_counter < num_images:  # Check if we still need more images
                        print(f'output: {outputs[i]}')
                        visualize(images[i], targets[i], outputs[i], output_dir, image_counter)
                        image_counter += 1
                    else:
                        break

    return val_loss / len(data_loader)

def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close('all')

def visualize(image, target, output, output_dir, index):

    # Tensor to PIL
    image_np = image.cpu().detach().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    
    # Ground truth
    gt_img = pil_image.copy()
    draw = ImageDraw.Draw(gt_img, 'RGBA')
    for box, label, mask in zip(target['boxes'], target['labels'], target['masks']):
        box = box.cpu().numpy().astype(int)
        mask = mask.cpu().numpy()
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=2)
        draw.text((box[0], box[1]), f"{label.item()}", fill='red')

    # Predictions
    pred_img = pil_image.copy()
    draw = ImageDraw.Draw(pred_img, 'RGBA')
    for box, label, score, mask in zip(output['boxes'], output['labels'], output['scores'], output['masks']):
        box = box.cpu().numpy().astype(int)
        label = label.item()
        score = score.item()
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='blue', width=2)
        draw.text((box[0], box[1]), f"{label}, {score:.2f}", fill='blue')

    # Combine images side by side
    combined = Image.new('RGB', (pil_image.width * 3, pil_image.height))
    combined.paste(pil_image, (0, 0))
    combined.paste(gt_img, (pil_image.width, 0))
    combined.paste(pred_img, (pil_image.width * 2, 0))

    combined.save(os.path.join(output_dir, f'visualization_{index}.png'))

    plt.close('all')

def transform_targets(targets):
    category_ids = targets['labels']
    bounding_boxes = targets['boxes']
    masks = targets['masks']

    transformed_targets = []

    for idx in range(len(category_ids)):
        
        target = {}
        
        target["boxes"] = bounding_boxes[idx]  
        if category_ids[idx].dim() > 0 and category_ids[idx].shape[0] == 1:
            target["labels"] = category_ids[idx]
        else:
            target["labels"] = category_ids[idx]
        target["masks"] = masks[idx]

        transformed_targets.append(target)
    
    return transformed_targets