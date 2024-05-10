import os
import torch
import numpy as np
from tqdm import tqdm

from model import get_model
from config import get_config
from dataloader import get_dataloader
from utils import plot_losses, train_one_epoch, evaluate

def main():

    # Get configurations
    args = get_config()
    os.environ['TORCH_HOME'] = '/mnt/ssd_4tb_0/huzaifa/models/maskrcnn'

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device(args.device)

    # Data setup
    train_loader = get_dataloader(os.path.join(args.data_dir,'train'), args.batch_size)
    val_loader = get_dataloader(os.path.join(args.data_dir,'valid'), args.batch_size)

    # Model setup
    model = get_model()
    model = model.to(device)

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Lists to collect train/validation loss
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training and validation loop
    for epoch in tqdm(range(args.num_epochs)):
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        val_loss = evaluate(model, val_loader, device, epoch, args.num_epochs, args.output_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

        val_loss = evaluate(model, val_loader, args.device, epoch, args.num_epochs, args.output_dir, num_images=3)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    plot_losses(train_losses, val_losses, args.output_dir)

if __name__ == "__main__":
    main()