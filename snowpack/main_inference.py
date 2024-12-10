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

from snowpack.dataset.data_utils import load_tifs_resize_to_np, load_tifs_resize_to_np_retain_ratio, recombine_patches, generate_final_mask, generate_patch_coords
from snowpack.main_finetune import get_dynamic_tiled_dataset, get_full_image_dataset

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
    "--path_to_config", type=str, default="configs/multiclass_boundary.json", help="hyperparameter configuration"
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")
parser.add_argument(
    "--test_path", type=str, default="snowpack/data/multiclass_10_2_boundary_pixels"
)
parser.add_argument(
    "--finetuning_checkpoint", type=str, default="", help="Finetuning checkpoint. Only used when zero shot is false"
)
parser.add_argument(
    "--zero_shot", action="store_true"
)
parser.add_argument(
    "--prompt_type", default="none", choices=["points", "none"]
)
parser.add_argument(
    "--full_image", default=True, type=bool
)

def main():
    args = parser.parse_args()

    with open(args.path_to_config) as f:
        cfg = json.load(f)

    if args.gpu != "cpu":
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if cfg['chunking']:
        _, test_dataset = get_dynamic_tiled_dataset(cfg, test_path=f'{args.data_path}test/')
    else:
        _, test_dataset = get_full_image_dataset(cfg, test_image_path=f'{args.data_path}test/images/',
                                            test_mask_path=f'{args.data_path}test/masks/',
                                            )

    main_worker(test_dataset=test_dataset, config=cfg)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epoch_mean_iou = []
    for idx, tup in enumerate(test_loader):
        image = np.array(tup[0].squeeze(0))
        mask = np.array(tup[1].squeeze(0))

        with torch.no_grad():
            predictor.set_image(image)

            # Generate embeddings (can be skipped for multiclass if prompts are not used)
            sparse_embeddings = torch.zeros((1, 1, 256), device=predictor.model.device)
            dense_prompt_embeddings = torch.zeros((1, 256, 256), device=predictor.model.device)
            dense_embeddings = F.interpolate(
                dense_prompt_embeddings.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Obtain predictions
            batched_mode = False
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

            low_res_masks = predictor.model(
                sparse_embeddings, dense_embeddings, high_res_features,
                predictor._features["image_embed"][-1].unsqueeze(0)
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]).float()

            # Prepare ground truth mask
            gt_mask = torch.tensor(mask, dtype=torch.long).to(device)

            # IoU computation
            pred_labels = torch.argmax(prd_masks, dim=1)  # Shape: [batch_size, H, W]

            iou_per_class = []
            for cls in range(prd_masks.shape[1]):  # Loop over classes
                inter = ((pred_labels == cls) & (gt_mask == cls)).sum()
                union = ((pred_labels == cls) | (gt_mask == cls)).sum()
                if union > 0:
                    iou_per_class.append((inter / union).item())

            mean_iou = np.mean(iou_per_class) if iou_per_class else 0
            epoch_mean_iou.append(mean_iou)

            print(f"Validation Mean IoU: {mean_iou}")

        if args.full_image:
            visualize_full_image_result(image, mask, prd_masks, store_pref, idx)
        else:
            print("Chunking visualization not yet implemented!")

"""
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
                if mask.sum() == 0:
                    print(f"Skipping mask at index {i} due to zero sum.")
                    continue

                if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                    continue

            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
            seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
            occupancy_mask[mask_bool] = True  # Update occupancy_mask

        else:
            seg_map = np.transpose(masks, (1, 2, 0))
"""

def visualize_full_image_result(image, gt, pred_mask, store_pref, idx):
    # Visualization: Show the original image, mask, and final segmentation side by side
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Reference Mask')
    plt.imshow(np.squeeze(gt), cmap='gray')
    plt.axis('off')

    #if args.prompt_type == "points":
    #     # Plot points in different colors
    #     colors = list(mcolors.TABLEAU_COLORS.values())
    #     for i, point in enumerate(np.squeeze(input_prompt)):
    #         plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()

    os.makedirs("prediction_results/", exist_ok=True)
    plt.savefig(f"prediction_results/{store_pref}_{idx}.png")

    #import cv2
    #gray_image = cv2.cvtColor(seg_map, cv2.COLOR_BGR2GRAY)
    #plt.imsave(f"grayed{j}.png", seg_map, cmap='gray')


def visualize_val_result(config, dataset, model):
    # Example
    image_size = (512, 512)
    patch_size = 256
    overlap = 128

    patch_coords = generate_patch_coords(image_size, patch_size, overlap)
    print(f"Patch coordinates: {patch_coords}")

    # Example inputs
    patch_predictions = [torch.rand(21, 256, 256) for _ in range(4)]  # Simulated 4 patches with 21 class probabilities
    patch_coords = [(0, 0), (128, 0), (0, 128), (128, 128)]  # Top-left corners of patches
    image_size = (384, 384)  # Original image size

    # Recombine patches
    combined_probabilities = recombine_patches(patch_predictions, patch_coords, image_size)

    # Generate final mask
    final_mask = generate_final_mask(combined_probabilities)

    print(f"Final mask shape: {final_mask.shape}")  # Should match the original image size (384, 384)


if __name__ == "__main__":
    main()