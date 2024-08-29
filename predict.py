from pathlib import Path
import numpy as np
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.beetlemodel import beetleResNet50
from datasets.beetledataset import beetleIndividualDataset



def main():

    val_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_val')
    gt_file_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images.csv')
    weights_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/results/weights/resnet50')
    save_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/results/prediction')

    weight_name = 'best_model.pth'
    # weight_name = 'epoch_10.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = beetleResNet50()
    model.load_state_dict(torch.load(weights_dir.joinpath(weight_name)))
    model.to(device)

    bs = 32

    val_dataset = beetleIndividualDataset(val_dir, gt_file_path)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        model.eval()

        total_loss = 0
        for batch in val_loader:
            image_name = batch['image_name']
            image, gt = batch['image'].to(device), batch['gt'].to(device)
            pred = model(image)

            val_loss = criterion(pred, gt)
            total_loss += val_loss.item()

            for i in range(image.size(0)):
                img = cv2.imread(str(val_dir.joinpath(image_name[i])))

                h, w, _ = img.shape
                # # Pad the image to square add zero padding to the right or bottom
                max_size = max(h, w)
                # if w > h:
                #     pad = np.zeros((w, w, 3), dtype=np.uint8)
                #     pad[:h, :, :] = img
                #     img = pad
                # else:
                #     pad = np.zeros((h, h, 3), dtype=np.uint8)
                #     pad[:, :w, :] = img
                #     img = pad

                # Resize the image to 224x224
                # img = cv2.resize(img, (224, 224))

                gt_coords = gt[i].cpu().numpy()
                gt_coords_int = gt_coords * max_size

                x1_gt, y1_gt, x2_gt, y2_gt = map(int, gt_coords_int[:4])
                x3_gt, y3_gt, x4_gt, y4_gt = map(int, gt_coords_int[4:])

                img = cv2.line(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 2)
                img = cv2.line(img, (x3_gt, y3_gt), (x4_gt, y4_gt), (0, 255, 0), 2)

                img = cv2.circle(img, (x1_gt, y1_gt), 5, (0, 255, 0), -1)
                img = cv2.circle(img, (x2_gt, y2_gt), 5, (0, 255, 0), -1)
                img = cv2.circle(img, (x3_gt, y3_gt), 5, (0, 255, 0), -1)
                img = cv2.circle(img, (x4_gt, y4_gt), 5, (0, 255, 0), -1)

                pred_coords = pred[i].cpu().numpy()
                # pred_coords_int = pred_coords
                pred_coords_int = pred_coords * max_size

                x1, y1, x2, y2 = map(int, pred_coords_int[:4])
                x3, y3, x4, y4 = map(int, pred_coords_int[4:])

                # print(x1, y1, x2, y2, x3, y3, x4, y4)

                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                img = cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)

                img = cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)
                img = cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1)
                img = cv2.circle(img, (x3, y3), 5, (0, 0, 255), -1)
                img = cv2.circle(img, (x4, y4), 5, (0, 0, 255), -1)

                save_path = save_dir.joinpath(image_name[i])
                cv2.imwrite(str(save_path), img)
                # exit()

        print(f'Val Loss: {total_loss / len(val_loader)}')

if __name__ == '__main__':
    main()