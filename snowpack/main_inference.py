import argparse
import warnings
import numpy as np
import os
import json

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from .dataset.data_utils import load_tifs_resize_to_np, load_tifs_resize_to_np_retain_ratio
from .dataset.dataset import SnowDataset

from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

parser.add_argument(
    "--pref", type=str, default="default", help="If not default, overwrites the storage name."
)
parser.add_argument(
    "--path_to_config", type=str, default="configs/revert_boundary_resize_simple.json", help="hyperparameter configuration"
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")
parser.add_argument(
    "--test_image_path", type=str, default="snowpack/data/images_test/"
)
parser.add_argument(
    "--test_mask_path", type=str, default="snowpack/data/masks_test/"
)
parser.add_argument(
    "--finetuning_checkpoint", type=str, default="", help="Finetuning checkpoint. Only used when zero shot is false"
)
parser.add_argument(
    "--zero_shot", action="store_true"
)
parser.add_argument(
    "--prompt_type", default="points", choices=["points", "none"]
)

def main():
    args = parser.parse_args()

    with open(args.path_to_config) as f:
        cfg = json.load(f)

    #set_seed(args.seed)

    if args.gpu != "cpu":
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if cfg['resize_method'] == "resize_retain_aspect":
        test_images, test_masks = load_tifs_resize_to_np_retain_ratio(args.test_image_path, args.test_mask_path)
    elif cfg['resize_method'] == "resize_simple":
        test_images, test_masks = load_tifs_resize_to_np(args.test_image_path, args.test_mask_path)
    else:
        raise NotImplementedError

    test_transforms = None

    # dataset setup
    test_ds = SnowDataset(test_images, test_masks, transforms=test_transforms, mask_type=cfg['mask_type'])

    main_worker(test_dataset=test_ds, config=cfg)

def main_worker(test_dataset, config):
    args = parser.parse_args()

    checkpoint_pref = args.path_to_config.split('/')[-1].split('.')[0]

    if args.pref == "default":
        store_pref = args.path_to_config.split('/')[-1].split('.')[0]
    else:
        store_pref = args.pref

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Load the fine-tuned model
    sam2_checkpoint = "snowpack/model/model_checkpoints/sam2.1_hiera_small.pt"
    #with resources.open_text('snowpack', ) as file:
    model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)

    if not args.zero_shot:
        print("Using finetuned weights.")
        saved_epoch = config['num_epochs']
        FINE_TUNED_MODEL_WEIGHTS = f"snowpack/model/model_checkpoints/snowpack_sam2_{checkpoint_pref}_{saved_epoch}.torch"
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
    else:
        print("Running inference zero shot.")

    # Perform inference and predict masks
    for j, tup in enumerate(test_loader):
        image = np.array(tup[0].squeeze(0))
        orig_mask = np.array(tup[1].squeeze(0))

        if args.prompt_type == "points":
            input_prompt = np.array(tup[2].squeeze(0))
            point_labels = np.ones([input_prompt.shape[0], 1])
        else:
            input_prompt = None
            point_labels = None

        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=input_prompt,
                point_labels=point_labels
            )

        if args.prompt_type == "points":
            # Process the predicted masks and sort by scores
            np_masks = np.array(masks[:, 0])
            np_scores = scores[:, 0]
            sorted_masks = np_masks[np.argsort(np_scores)][::-1]

            # Initialize segmentation map and occupancy mask
            seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
            occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

            # Combine masks to create the final segmentation map
            for i in range(sorted_masks.shape[0]):
                mask = sorted_masks[i]
                if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                    continue

            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
            seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
            occupancy_mask[mask_bool] = True  # Update occupancy_mask

        else:
            seg_map = np.transpose(masks, (1, 2, 0))

        # Visualization: Show the original image, mask, and final segmentation side by side
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.title('Test Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Reference Mask and Prompts')
        plt.imshow(np.squeeze(orig_mask), cmap='gray')
        plt.axis('off')

        if args.prompt_type == "points":
            # Plot points in different colors
            colors = list(mcolors.TABLEAU_COLORS.values())
            for i, point in enumerate(np.squeeze(input_prompt)):
                plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')

        plt.subplot(1, 3, 3)
        plt.title('Final Segmentation')
        plt.imshow(seg_map, cmap='jet')
        plt.axis('off')

        plt.tight_layout()

        os.makedirs("prediction_results/", exist_ok=True)
        plt.savefig(f"prediction_results/{store_pref}_{j}.png")
                

if __name__ == "__main__":
    main()