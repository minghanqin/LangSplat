import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import colormaps
from pathlib import Path


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='firebrick', marker='o',
               s=marker_size, edgecolor='black', linewidth=2.5, alpha=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o',
               s=marker_size, edgecolor='black', linewidth=1.5, alpha=1)   


def show_box(boxes, ax, color=None):
    if type(color) == str and color == 'random':
        color = np.random.random(3)
    elif color is None:
        color = 'black'
    for box in boxes.reshape(-1, 4):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=4, 
                                   capstyle='round', joinstyle='round', linestyle='dotted')) 


def show_result(image, point, bbox, save_path):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle((0, 0), image.shape[1]-1, image.shape[0]-1, linewidth=0, edgecolor='none', facecolor='white', alpha=0.3)
    plt.gca().add_patch(rect)
    input_point = point.reshape(1,-1)
    input_label = np.array([1])
    show_points(input_point, input_label, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()


def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth


def colormap_saving(image: torch.Tensor, colormap_options, save_path):
    """
    if image's shape is (h, w, 1): draw colored relevance map;
    if image's shape is (h, w, 3): return directively;
    if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )
    if save_path is not None:
        media.write_image(save_path.with_suffix(".png"), output_image, fmt="png")
    return output_image


def vis_mask_save(mask, save_path: Path = None):
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path), mask_save)


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask