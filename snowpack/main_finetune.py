import argparse
import warnings
import numpy as np
import json
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# from .dataset.data_utils import load_tifs_resize_to_np, load_tifs_resize_to_np_retain_ratio
# from .dataset.dataset import SnowDataset

from snowpack.dataset.data_utils import load_tifs_resize_to_np, load_tifs_resize_to_np_retain_ratio
from snowpack.dataset.dataset import SnowDataset
from snowpack.dataset.dynamic_tiled_dataset import DynamicImagePatchesDataset
from snowpack.train_epochs import *
from snowpack.dataset.augmentation import get_transformation

from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


from tqdm import tqdm, trange
from collections import Counter
import cv2

import wandb

from sklearn.model_selection import KFold
import copy


parser = argparse.ArgumentParser(description="PyTorch Unet Training")

parser.add_argument(
    "--path_to_config", type=str, default="configs/revert_boundary_resize_simple.json", help="hyperparameter configuration"
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")

parser.add_argument(
    "--data_path", type=str, default="snowpack/data/multiclass_10_2/" # this one has 20 classes
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

parser.add_argument(
    '--do_kfold', 
    default=False, 
    action='store_true', 
    help='do kfold cross validation'
)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
parser.add_argument(
    '--multiclass', 
    default=False, 
    action='store_true', 
    help='do multiclass training'
)
parser.add_argument('--n_classes', default=20, type=int)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def main():
    args = parser.parse_args()

    pref = args.path_to_config.split('/')[-1].split('.')[0]

    if args.use_wandb:
        wandb.login()

    with open(args.path_to_config) as f:
        config = json.load(f)

    if args.use_wandb and args is not None:
        wandb.init(
            entity='sea-ice',
            project='snowpack',
            name=pref,
        )
        wandb.config.update(config)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: binarize / erode masks (so that we don't have points that are too close to the border)
    # Note: the eroded mask only seems to be used for prompt generation (slightly reduces mask size)
    # (do we have noisy boundaries?)
    # Prompts: https://www.datacamp.com/tutorial/sam2-fine-tuning
    # TODO: Visualization of the selected points / prompts

    NUM_EPOCHS = config['num_epochs']
    NUM_EPOCHS_K_FOLD = 20  # idk what this should be
    FINETUNED_MODEL_NAME = "snowpack_sam2"
    NUM_K_FOLDS = 3 # should be higher with a bigger batch size
    # Very lazy config updates:
    config['chunking'] = True
    config['learning_rate'] = config['learning_rate'] * 100
    if args.multiclass:
        config['mask_type'] = 'layer'
        print(f'Currently doing multiclass with {args.n_classes} classes. Need to have data folder location match up too')
        
    class_weights = get_class_weights(args.data_path, device) if args.multiclass else None
    # TODO: NOTE: I multiplied the learning rate by 100, might want that/might not want that
    # also scheduler is now different and idk if that's good tbh (older/original version is commented out)

    accumulation_steps = 7

    if args.do_kfold:
        # train here has to include everything
        if config['chunking']:
            dataset = get_dynamic_tiled_dataset(config, args, train_path=f'{args.data_path}train/')
        else:
            dataset = get_full_image_dataset(config, args, train_image_path=f'{args.data_path}train/images/',
                                                train_mask_path=f'{args.data_path}train/masks/',
                                                )
        k_fold(args, config, dataset, accumulation_steps, NUM_EPOCHS_K_FOLD, device, pref, class_weights, k=NUM_K_FOLDS)

    else:
        if config['chunking']:
            train_dataset, test_dataset = get_dynamic_tiled_dataset(config, args, train_path=f'{args.data_path}train/',
                                                                    test_path=f'{args.data_path}test/'
                                                                    )
        else:
            train_dataset, test_dataset = get_full_image_dataset(config, args, train_image_path=f'{args.data_path}train/images/',
                                                                    train_mask_path=f'{args.data_path}train/masks/',
                                                                    test_image_path=f'{args.data_path}test/images/',
                                                                    test_mask_path=f'{args.data_path}test/masks/'
                                                                    )
        regular_train(args, config, train_dataset, test_dataset, accumulation_steps, FINETUNED_MODEL_NAME, NUM_EPOCHS, device, pref, class_weights)



def get_model_optimizer_scaler_scheduler(args, config, device):
    # model setup
    sam2_checkpoint = "snowpack/model/model_checkpoints/sam2.1_hiera_small.pt"
    #with resources.open_text('snowpack', ) as file:
    model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    sam2_model = build_sam2(config_file=model_cfg, ckpt_path=sam2_checkpoint, device="cuda")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if args.multiclass:
        sam2_model = MulticlassSAMWrapper(sam2_model, args.n_classes).to(device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    predictor = SAM2ImagePredictor(sam2_model)

    # make prompt encoder and mask decoder trainable
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    predictor.model.sam_mask_decoder = predictor.model.sam_mask_decoder.to(torch.float16)
    predictor.model.sam_prompt_encoder = predictor.model.sam_prompt_encoder.to(torch.float16)

    # training setup
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=config['learning_rate'],weight_decay=1e-4) #1e-5, weight_decay = 4e-5
    # mix precision
    scaler = torch.amp.GradScaler(device.type)
    # wandb setup
    if args.use_wandb and not args.do_kfold:
        wandb.watch(predictor.model, log_freq=16)

    # Initialize scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # 500 , 250, gamma = 0.1

    return predictor, optimizer, scaler, scheduler



def get_full_image_dataset(cfg, args, train_image_path=None, 
                           train_mask_path=None,
                           test_image_path=None, 
                           test_mask_path=None,
                           train_transforms=None,
                           test_transforms=None
                           ):

    if cfg['resize_method'] == "resize_retain_aspect":
        train_images, train_masks = load_tifs_resize_to_np_retain_ratio(train_image_path, train_mask_path)
    elif cfg['resize_method'] == "resize_simple":
        train_images, train_masks = load_tifs_resize_to_np(train_image_path, train_mask_path)
    else:
        raise NotImplementedError
    # dataset setup
    train_dataset = SnowDataset(train_images, train_masks, transforms=train_transforms, mask_type=cfg['mask_type'], 
                                size_strategy=cfg['resize_method'], dilate=cfg['dilate'])
    if not test_image_path:
        return train_dataset

    if cfg['resize_method'] == "resize_retain_aspect":
        test_images, test_masks = load_tifs_resize_to_np_retain_ratio(test_image_path, test_mask_path)
    elif cfg['resize_method'] == "resize_simple":
        test_images, test_masks = load_tifs_resize_to_np(test_image_path, test_mask_path)
    else:
        raise NotImplementedError
    # dataset setup
    if args.multiclass:
        test_dataset = SnowDataset(test_images, test_masks, transforms=test_transforms, dilate=cfg['dilate'], mask_type=cfg['mask_type'])
    else:
        test_dataset = SnowDataset(test_images, test_masks, transforms=test_transforms, dilate=cfg['dilate'])

    return train_dataset, test_dataset


def get_dynamic_tiled_dataset(cfg=None, 
                              args=None,
                              train_path=None,
                              test_path=None
                              ):
    
    # TODO: messy here, move this into the config
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    boundary_mask = False
    revert = True
    dilate = True
    train_transforms, test_transforms = get_transformation(mean=mean, std=std)
    #train_transforms, test_transforms = None, None
    patch_size = 1024
    overlap = 0
    kernel_size=20

    print("Uses chunking")

    train_ds = DynamicImagePatchesDataset(data_dir=train_path, transform=train_transforms, patch_size=patch_size, overlap=overlap, inference_mode=False, boundary_mask=boundary_mask, revert=revert, dilate=dilate, kernel_size=kernel_size)

    if not test_path:
        return train_ds

    test_ds = DynamicImagePatchesDataset(data_dir=test_path, transform=test_transforms, patch_size=patch_size, overlap=overlap, inference_mode=False, boundary_mask=boundary_mask, revert=revert, dilate=dilate, kernel_size=kernel_size)

    return train_ds, test_ds


def regular_train(args, cfg, train_dataset, test_dataset, accumulation_steps, 
                  FINETUNED_MODEL_NAME, NUM_EPOCHS, device, pref, class_weights,
                  num_workers=1, first_class_is_1=True):

    train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=num_workers,
    )

    val_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers
    )

    predictor, optimizer, scaler, scheduler = get_model_optimizer_scaler_scheduler(args, cfg, device)

    for epoch in trange(1, NUM_EPOCHS + 1):
        if args.multiclass:
            mean_train_iou, loss = multiclass_epoch(train_loader, predictor, accumulation_steps, epoch, 
                scheduler, scaler, optimizer, device, class_weights, args, first_class_is_1=False)
            with torch.amp.autocast(device.type):
                mean_test_iou = validate_multiclass(val_loader, predictor, epoch, device, args)

        else:
            mean_train_iou, loss = binary_epoch(train_loader, predictor, accumulation_steps, epoch, 
                scheduler, scaler, optimizer, device, class_weights, args)
            with torch.amp.autocast(device.type):
                mean_test_iou = validate_binary(val_loader, predictor, epoch, device, args)


    FINETUNED_MODEL = FINETUNED_MODEL_NAME + "_" + str(pref) + "_" + str(epoch) + ".torch"
    torch.save(predictor.model.state_dict(), os.path.join("snowpack/model/model_checkpoints", FINETUNED_MODEL))



def k_fold(args, cfg, dataset, accumulation_steps, NUM_EPOCHS, device, pref, class_weights, k, num_workers=1, first_class_is_1=True):
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    
    fold_results = []  # To store results for each fold
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(dataset))):
        print(f"Starting Fold {fold + 1}/{k}")
        
        # Split dataset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Create dataloaders
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=num_workers)
        
        # Reinitialize model, optimizer, and scheduler
        predictor, optimizer, scaler, scheduler = get_model_optimizer_scaler_scheduler(args, cfg, device)
        
        # Train and validate for the current fold
        # fold_metrics = train_and_validate(train_loader, val_loader, model, optimizer, scheduler, scaler, args, config)
        for epoch in trange(1, NUM_EPOCHS + 1):
            if args.multiclass:
                mean_train_iou, loss = multiclass_epoch(train_loader, predictor, accumulation_steps, epoch, 
                                                    scheduler, scaler, optimizer, device, class_weights, args, first_class_is_1)
                with torch.amp.autocast(device.type):
                    mean_val_iou = validate_multiclass(val_loader, predictor, epoch, device, args, first_class_is_1)

            else:
                mean_train_iou, loss = binary_epoch(train_loader, predictor, accumulation_steps, epoch, 
                                                    scheduler, scaler, optimizer, device, class_weights, args)
                with torch.amp.autocast(device.type):
                    mean_val_iou = validate_binary(val_loader, predictor, epoch, device, args)
        fold_metrics = {
            'final_train_iou': mean_train_iou,  # Replace with actual metrics from the training loop
            'final_val_iou': mean_val_iou,
            'loss': loss
            }
        if args.use_wandb:
            wandb.log({'kfold_fold#': fold, 
                       'kfold_train_iou': mean_train_iou, 
                       'kfold_val_iou': mean_val_iou, 
                       'kfold_loss': loss})
        print('Average metrics across curent fold:', fold_metrics)
        fold_results.append(fold_metrics)
    
    # Calculate average metrics across folds
    k_fold_metrics = {key: sum(f[key] for f in fold_results) / k for key in fold_results[0]}
    print('Average metrics across folds:', k_fold_metrics)
    torch.save(k_fold_metrics, 'snowpack/k_fold_metrics.pt')

def get_class_weights(data_path, device):
    class_pixel_counts = Counter()
    masks_train = [f'{data_path}train/masks/{i}' for i in os.listdir(f'{data_path}train/masks/') if 'store' not in i and i.endswith('.tiff')]
    masks_test = [f'{data_path}test/masks/{i}' for i in os.listdir(f'{data_path}test/masks/') if 'store' not in i and i.endswith('.tiff')]
    masks_train.extend(masks_test)
    masks_paths = masks_train

    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)        
        unique, counts = np.unique(mask, return_counts=True)
        class_pixel_counts.update(dict(zip(unique, counts)))

    total_pixels = sum(class_pixel_counts.values())

    # Compute weights as inverse of class frequency
    class_weights = {cls: total_pixels / count for cls, count in class_pixel_counts.items()}
    class_weights = dict(sorted(class_weights.items()))

    # Normalize weights (optional, for better scaling)
    normalized_weights = {cls: weight / max(class_weights.values()) for cls, weight in class_weights.items()}

    return torch.tensor(list((normalized_weights.values()))).to(device).to(torch.float32)

if __name__ == "__main__":
    main()