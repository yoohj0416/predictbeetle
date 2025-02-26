from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.beetlemodel import *
from beetledatasets.beetledataset import beetleIndividualDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_and_write(f, message):
    print(message)
    f.write(message + '\n')

def main():

    # Load arguments and config
    args = parse_args()
    config = load_config(args.config)

    # Print config
    print('Config:')
    for k, v in config.items():
        print(f'  {k}: {v}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_name = config['model']['name']
    model = eval(model_name)()
    model.to(device)

    # Load dataset
    dataset_dir = Path(config['dataset']['source_dir'])

    train_dataset = beetleIndividualDataset(
        dataset_dir.joinpath(config['dataset']['train']['images']),
        dataset_dir.joinpath(config['dataset']['train']['labels']),
        input_size=config['model']['input_size'],
        horizontal_flip=config['train']['augmentations']['horizontal_flip'],
        vertical_flip=config['train']['augmentations']['vertical_flip']
    )

    val_dataset = beetleIndividualDataset(
        dataset_dir.joinpath(config['dataset']['val']['images']),
        dataset_dir.joinpath(config['dataset']['val']['labels']),
        input_size=config['model']['input_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'], 
        shuffle=False
    )

    # Set criterion and optimizer
    criterion = nn.MSELoss(reduction=config['train']['mse_reduction'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create weights directory
    weights_dir = Path('results/weights')
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Make a directory to save train results
    save_dir = weights_dir.joinpath(Path(args.config).stem)
    if save_dir.exists():
        raise ValueError(f'{save_dir} already exists')
    save_dir.mkdir()

    # Save config to save_dir
    with open(save_dir.joinpath('config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Make a log file
    log_path = save_dir.joinpath('train.log')
    with open(log_path, 'w') as f:
        f.write(f'Start training at {time.ctime()}\n')

    best_val_loss = float('inf')
    epochs = config['train']['epochs']
    
    for epoch in range(epochs):
        # Train
        model.train()
        for i, batch in enumerate(train_loader):
            images, gts = batch['image'].to(device), batch['gt'].to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, gts)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                with open(log_path, 'a') as f:
                    print_and_write(f, f'Epoch {epoch + 1}/{epochs}, Iteration {i}/{len(train_loader)}, Train Loss: {loss.item()}')

        # Validate
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                images, gts = batch['image'].to(device), batch['gt'].to(device)
                preds = model(images)
                loss = criterion(preds, gts)
                total_loss += loss.item()

            with open(log_path, 'a') as f:
                print_and_write(f, f'Epoch {epoch + 1}/{epochs}, Val Loss: {total_loss / len(val_loader)}')

        # Save best model
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            torch.save(model.state_dict(), save_dir.joinpath('best_model.pth'))
            with open(log_path, 'a') as f:
                print_and_write(f, f'Save best model at epoch {epoch + 1}')

        # Save model every save_every epochs
        if (epoch + 1) % config['train']['save_every'] == 0:
            torch.save(model.state_dict(), save_dir.joinpath(f'epoch_{epoch + 1}.pth'))
            with open(log_path, 'a') as f:
                print_and_write(f, f'Save model at epoch {epoch + 1}')

    with open(log_path, 'a') as f:
        f.write(f'End training at {time.ctime()}\n')

if __name__ == '__main__':
    main()