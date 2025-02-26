from matplotlib import pyplot as plt
import numpy as np
import cv2

def visualize_gt_pred(image_name, gt, pred, from_dir, save_dir, dpi=100):

    # Parameters for visualization
    line_width = 4
    circle_size = 9
    x_marker_size = 100
    x_marker_width = 3

    # Load image
    img = cv2.imread(str(from_dir.joinpath(image_name)))
    if img is None:
        print(f"Image {image_name} not found in {from_dir}")
        return
    h, w, _ = img.shape
    max_size = max(h, w)

    # Transform gt and pred to coordinates
    gt_coords = gt.cpu().numpy()
    gt_coords_int = gt_coords * max_size
    x1_gt, y1_gt, x2_gt, y2_gt = map(int, gt_coords_int[:4])
    x3_gt, y3_gt, x4_gt, y4_gt = map(int, gt_coords_int[4:])

    pred_coords = pred.cpu().numpy()
    pred_coords_int = pred_coords * max_size
    x1, y1, x2, y2 = map(int, pred_coords_int[:4])
    x3, y3, x4, y4 = map(int, pred_coords_int[4:])

    # Make figure
    fig = plt.figure(frameon=False, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Draw image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Draw ground truth
    ax.plot([x1_gt, x2_gt], [y1_gt, y2_gt], color='green', linewidth=line_width)
    ax.plot([x3_gt, x4_gt], [y3_gt, y4_gt], color='green', linewidth=line_width)
    ax.plot(x1_gt, y1_gt, 'go', markersize=circle_size)
    ax.plot(x2_gt, y2_gt, 'go', markersize=circle_size)
    ax.plot(x3_gt, y3_gt, 'go', markersize=circle_size)
    ax.plot(x4_gt, y4_gt, 'go', markersize=circle_size)

    # Draw prediction
    ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', linewidth=line_width)
    ax.plot([x3, x4], [y3, y4], color='red', linestyle='--', linewidth=line_width)
    ax.scatter(x1, y1, color='red', marker='x', s=x_marker_size, linewidths=x_marker_width)
    ax.scatter(x2, y2, color='red', marker='x', s=x_marker_size, linewidths=x_marker_width)
    ax.scatter(x3, y3, color='red', marker='x', s=x_marker_size, linewidths=x_marker_width)
    ax.scatter(x4, y4, color='red', marker='x', s=x_marker_size, linewidths=x_marker_width)

    # Save image and close plot
    fig.savefig(str(save_dir.joinpath(image_name)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
