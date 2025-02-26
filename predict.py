from pathlib import Path
import numpy as np
import cv2
import argparse
import yaml
from tqdm import tqdm
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.beetlemodel import *
from beetledatasets.beetledataset import beetleIndividualDataset

from utils.visualize import visualize_gt_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_coord', action='store_true')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_pred_coords(image_name, pred, from_dir, coords_path):
    
    # Load image
    img = cv2.imread(str(from_dir.joinpath(image_name)))
    if img is None:
        print(f"Image {image_name} not found in {from_dir}")
        return
    h, w, _ = img.shape
    max_size = max(h, w)

    # Transform pred to coordinates
    pred_coords = pred.cpu().numpy()
    pred_coords_int = pred_coords * max_size
    x1, y1, x2, y2 = map(int, pred_coords_int[:4])
    x3, y3, x4, y4 = map(int, pred_coords_int[4:])

    # Save coordinates
    with open(coords_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, (x1, y1, x2, y2), (x3, y3, x4, y4)])

def main():

    # Load arguments and config
    args = parse_args()
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_name = config['model']['name']
    model = eval(model_name)()
    pretrained_path = Path('results/weights').joinpath(Path(args.config).stem, config['test']['weights'])
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    # Load dataset
    dataset_dir = Path(config['dataset']['source_dir'])

    test_dataset = beetleIndividualDataset(
        dataset_dir.joinpath(config['dataset']['test']['images']),
        dataset_dir.joinpath(config['dataset']['test']['labels']),
        input_size=config['model']['input_size']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test']['batch_size'],
        shuffle=False
    )

    # Set criterion
    criterion = nn.MSELoss(reduction=config['test']['mse_reduction'])

    # Set save visualization directory
    if config['test']['visualize']:
        save_dir = Path('results/predictions').joinpath(Path(args.config).stem, 'images')
        if save_dir.exists():
            raise FileExistsError(f'{save_dir} already exists')
        save_dir.mkdir(parents=True)
        from_dir = dataset_dir.joinpath(config['dataset']['test']['images'])

    if args.save_coord:
        coords_path = Path('results/predictions').joinpath(Path(args.config).stem, 'pred_coords.csv')
        with open(coords_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BeetleID', 'coords_len', 'coords_width'])
        from_dir = dataset_dir.joinpath(config['dataset']['test']['images'])

    # Inference
    with torch.no_grad():
        model.eval()

        total_loss = 0
        for batch in tqdm(test_loader):
            image_names = batch['image_name']
            images, gts = batch['image'].to(device), batch['gt'].to(device)
            preds = model(images)

            val_loss = criterion(preds, gts)
            total_loss += val_loss.item()

            if config['test']['visualize']:
                for i in range(images.size(0)):
                    visualize_gt_pred(image_names[i], gts[i], preds[i], from_dir, save_dir)

            if args.save_coord:
                for i in range(images.size(0)):
                    save_pred_coords(image_names[i], preds[i], from_dir, coords_path)

        print(f'MSE Loss: {total_loss / len(test_loader)}')

if __name__ == '__main__':
    main()