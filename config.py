import torch
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Configurations for Mask R-CNN')

    parser.add_argument('--data_dir', type=str, default='/mnt/ssd_4tb_0/huzaifa/oatmeal/braintumor_sampleset', help='Path to the data')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/huzaifa/workspace/oatmeal-outputs/checkpoints', help='Path to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='/home/huzaifa/workspace/oatmeal-outputs/outputs', help='Path to save loss plot and visualizations')

    parser.add_argument('--input_size', type=int, default=800, help='Input size of the image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for Adam optimizer')

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')

    args = parser.parse_args()
    return args