import argparse
import warnings
import numpy as np
import json
import os

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

import wandb

wandb.login()

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

parser.add_argument(
    "--path_to_config", type=str, default="configs/revert_boundary_resize_simple.json", help="hyperparameter configuration"
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")
parser.add_argument(
    "--train_image_path", type=str, default="snowpack/data/images_train/"
)
parser.add_argument(
    "--train_mask_path", type=str, default="snowpack/data/masks_train/"
)
parser.add_argument(
    "--test_image_path", type=str, default="snowpack/data/images_test/"
)
parser.add_argument(
    "--test_mask_path", type=str, default="snowpack/data/masks_test/"
)
parser.add_argument(
    '--use_wandb', 
    default=True, 
    action='store_false', 
    help='use wandb for logging'
)
parser.add_argument(
    '--wandb_entity',
    type=str,
    default="sea-ice"
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
parser.add_argument('--multiclass', default=False, type=bool) ###########
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
        else:
            print("CUDA is not available.")

    if cfg['resize_method'] == "resize_retain_aspect":
        train_images, train_masks = load_tifs_resize_to_np_retain_ratio(args.train_image_path, args.train_mask_path)
        test_images, test_masks = load_tifs_resize_to_np_retain_ratio(args.test_image_path, args.test_mask_path)
    elif cfg['resize_method'] == "resize_simple":
        train_images, train_masks = load_tifs_resize_to_np(args.train_image_path, args.train_mask_path)
        test_images, test_masks = load_tifs_resize_to_np(args.test_image_path, args.test_mask_path)
    else:
        raise NotImplementedError

    train_transforms, test_transforms = None, None

    # dataset setup
    train_ds = SnowDataset(train_images, train_masks, transforms=train_transforms, mask_type=cfg['mask_type'], size_strategy=cfg['resize_method'], dilate=cfg['dilate'])
    test_ds = SnowDataset(test_images, test_masks, transforms=test_transforms, dilate=cfg['dilate'])

    main_worker(args, train_dataset=train_ds, test_dataset=test_ds, config=cfg)

def convert_to_binary_masks(mask, n_classes):
    binary_masks = torch.zeros(n_classes, mask.shape[-2], mask.shape[-1]).cuda()
    for cls in range(n_classes):
        binary_masks[cls] = (mask == cls).float()
    return binary_masks


def main_worker(args, train_dataset, test_dataset, config):
    # TODO: binarize / erode masks (so that we don't have points that are too close to the border)
    # Note: the eroded mask only seems to be used for prompt generation (slightly reduces mask size)
    # (do we have noisy boundaries?)
    # Prompts: https://www.datacamp.com/tutorial/sam2-fine-tuning
    # TODO: Visualization of the selected points / prompts

    pref = args.path_to_config.split('/')[-1].split('.')[0]

    NUM_EPOCHS = config['num_epochs']
    FINETUNED_MODEL_NAME = "snowpack_sam2"

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # model setup
    sam2_checkpoint = "snowpack/model/model_checkpoints/sam2.1_hiera_small.pt"
    #with resources.open_text('snowpack', ) as file:
    model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'

    sam2_model = build_sam2(config_file=model_cfg, ckpt_path=sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # make prompt encoder and mask decoder trainable
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # training setup
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=config['learning_rate'],weight_decay=1e-4) #1e-5, weight_decay = 4e-5
    # mix precision
    scaler = torch.cuda.amp.GradScaler()

    #torch.cuda.set_device(args.gpu)
    #sam2_model = sam2_model.cuda(args.gpu)
    #predictor = predictor.cuda(args.gpu)

    # wandb setup
    if args.use_wandb and args is not None:

        wandb.init(
            entity='sea-ice',
            project='snowpack',
            name=pref,
        )
        wandb.config.update(config)
        wandb.watch(predictor.model, log_freq=2)

#######################################################################################################################################
#######################################################################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if args.multiclass:
       class_weights = torch.load('weights_20_2.pt').cuda()
       n_classes = 40 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
    accumulation_steps = 4
    for epoch in range(1, NUM_EPOCHS + 1):
        predictor.model.train()  # Ensure the model is in training mode
        epoch_loss = 0.0  # Track the cumulative loss for the epoch

        for step, tup in enumerate(train_loader):
            image = np.array(tup[0].squeeze(0))
            mask = np.array(tup[1].squeeze(0))  # Ground truth mask
            input_prompt = np.array(tup[2].squeeze(0))
            num_masks = tup[3].squeeze(0)

            if image is None or mask is None or num_masks == 0:
                print("Continuing due to empty image, mask, or no masks", flush=True)
                continue

            predictor.set_image(image)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                if args.multiclass:
                    # Multiclass option
                    binary_gt_masks = convert_to_binary_masks(mask, n_classes)

                    binary_prd_masks = []
                    for cls in range(n_classes):
                        binary_mask = (predictor.model.sam_mask_decoder.predict(image) == cls).float()
                        binary_prd_masks.append(binary_mask)

                    binary_prd_masks = torch.stack(binary_prd_masks, dim=0)

                    total_loss = 0
                    for cls in range(n_classes):
                        loss_per_class = F.binary_cross_entropy_with_logits(
                            binary_prd_masks[cls], binary_gt_masks[cls]
                        )
                        total_loss += loss_per_class

                    total_loss /= n_classes

                else:
                    # Binary option
                    gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                    prd_mask = torch.sigmoid(predictor.model.sam_mask_decoder.predict(image))
                    total_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        # Scheduler step after epoch
        scheduler.step()

        # Log epoch-level metrics
        wandb.log({"epoch": epoch, "epoch_train_loss": epoch_loss / len(train_loader)})
        print(f"Epoch {epoch}/{NUM_EPOCHS}: Train Loss = {epoch_loss / len(train_loader):.4f}")

                        

        # Validate
        predictor.model.eval()  # Ensure the model is in evaluation mode
        val_loss = 0.0  # Track cumulative validation loss
        iou_scores = []  # Store IoU for each class or binary IoU

        with torch.no_grad():
            for step, tup in enumerate(val_loader):
                image = np.array(tup[0].squeeze(0))
                mask = np.array(tup[1].squeeze(0))  # Ground truth mask
                input_prompt = np.array(tup[2].squeeze(0))
                num_masks = tup[3].squeeze(0)

                if image is None or mask is None or num_masks == 0:
                    print("Skipping validation due to empty data", flush=True)
                    continue

                predictor.set_image(image)

                if args.multiclass:
                    # Multiclass validation
                    binary_gt_masks = convert_to_binary_masks(mask, n_classes)

                    binary_prd_masks = []
                    for cls in range(n_classes):
                        binary_mask = (predictor.model.sam_mask_decoder.predict(image) == cls).float()
                        binary_prd_masks.append(binary_mask)

                    binary_prd_masks = torch.stack(binary_prd_masks, dim=0)

                    total_loss = 0
                    iou_per_class = []

                    for cls in range(n_classes):
                        loss_per_class = F.binary_cross_entropy_with_logits(
                            binary_prd_masks[cls], binary_gt_masks[cls]
                        )
                        total_loss += loss_per_class

                        # Calculate IoU for each class
                        pred_binary_mask = torch.sigmoid(binary_prd_masks[cls]) > 0.5
                        inter = (pred_binary_mask & binary_gt_masks[cls].bool()).sum().item()
                        union = (pred_binary_mask | binary_gt_masks[cls].bool()).sum().item()
                        if union > 0:
                            iou_per_class.append(inter / union)

                    total_loss /= n_classes
                    mean_iou = np.mean(iou_per_class) if iou_per_class else 0
                    iou_scores.extend(iou_per_class)  # Collect IoUs for all classes

                    # Log class-specific IoUs
                    for cls, iou in enumerate(iou_per_class):
                        wandb.log({f"val_iou_class_{cls}": iou})

                else:
                    # Binary validation
                    gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                    prd_mask = torch.sigmoid(predictor.model.sam_mask_decoder.predict(image))

                    total_loss = (-gt_mask * torch.log(prd_mask + 1e-6) -
                                (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()

                    inter = (gt_mask * (prd_mask > 0.5)).sum()
                    union = (gt_mask + (prd_mask > 0.5) - (gt_mask * (prd_mask > 0.5))).sum()
                    iou = inter / union if union > 0 else 0
                    iou_scores.append(iou.item())

                val_loss += total_loss.item()

        # Compute final validation metrics
        avg_val_loss = val_loss / len(val_loader)
        overall_mean_iou = np.mean(iou_scores) if iou_scores else 0

        # Log final validation metrics
        wandb.log({
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "val_mean_iou": overall_mean_iou
        })

        print(f"Validation Loss: {avg_val_loss:.4f}, Mean IoU: {overall_mean_iou:.4f}")


    FINETUNED_MODEL = FINETUNED_MODEL_NAME + "_" + str(pref) + "_" + str(epoch) + ".torch"
    torch.save(predictor.model.state_dict(), os.path.join("snowpack/model/model_checkpoints", FINETUNED_MODEL))

if __name__ == "__main__":
    main()