from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.beetlemodel import beetleResNet50
from datasets.beetledataset import beetleIndividualDataset


def main():

    train_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_train')
    val_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_val')
    gt_file_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images.csv')
    weights_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/results/weights/resnet50')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bs = 32
    epochs = 100

    model = beetleResNet50()
    model.to(device)

    train_dataset = beetleIndividualDataset(train_dir, gt_file_path)
    val_dataset = beetleIndividualDataset(val_dir, gt_file_path)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            image, gt = batch['image'].to(device), batch['gt'].to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Iteration {i}/{len(train_loader)}, Train Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                image, gt = batch['image'].to(device), batch['gt'].to(device)
                pred = model(image)
                loss = criterion(pred, gt)
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {total_loss / len(val_loader)}')

        # Save best model every epoch and save 10th epoch model
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            torch.save(model.state_dict(), weights_dir.joinpath(f'best_model.pth'))
            print(f'Save best model at epoch {epoch + 1}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), weights_dir.joinpath(f'epoch_{epoch + 1}.pth'))
            print(f'Save model at epoch {epoch + 1}')


if __name__ == '__main__':
    main()