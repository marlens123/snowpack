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

from dataset.data_utils import load_tifs_resize_to_np, load_tifs_resize_to_np_retain_ratio
from dataset.dataset import SnowDataset

from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


from tqdm import tqdm, trange

import wandb

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

parser.add_argument(
    "--path_to_config", type=str, default="configs/revert_boundary_resize_simple.json", help="hyperparameter configuration"
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")
# parser.add_argument(
#     "--train_image_path", type=str, default="snowpack/data/images_train/"
# )
# parser.add_argument(
#     "--train_mask_path", type=str, default="snowpack/data/masks_train/"
# )
# parser.add_argument(
#     "--test_image_path", type=str, default="snowpack/data/images_test/"
# )
# parser.add_argument(
#     "--test_mask_path", type=str, default="snowpack/data/masks_test/"
# )

parser.add_argument(
    "--data_path", type=str, default="snowpack/data/multiclass/"
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
parser.add_argument('--nclasses', default=40, type=int) ########### black is no mask, I forgot |:

class MulticlassSAMWrapper(nn.Module):
    def __init__(self, sam_model, n_classes):
        super(MulticlassSAMWrapper, self).__init__()
        self.model = sam_model
        self.sam_mask_decoder = self.model.sam_mask_decoder
        self.sam_prompt_encoder = self.model.sam_prompt_encoder
        self.image_size = self.model.image_size #### we can probably change this
        self.no_mem_embed = self.model.no_mem_embed


        self.device = self.model.device

        self.multiclass_head = nn.Conv2d(
            in_channels=1, 
            out_channels=n_classes,
            kernel_size=1
        )

    def set_image(self, image):
        return self.model.set_image(image)
    
    def forward_image(self, image):
        return self.model.forward_image(image)
    
    def _prepare_backbone_features(self, features):
        return self.model._prepare_backbone_features(features)
    
    def directly_add_no_mem_embed(self, sparse_embeddings, dense_embeddings):
        return self.model.directly_add_no_mem_embed(sparse_embeddings, dense_embeddings)

    def forward(self, sparse_embeddings, dense_embeddings, high_res_features, feats):
        # Get image embeddings using forward_image
        # image_embeddings = self.forward_image(image)

        # # Decode masks with SAM's mask decoder
        # low_res_masks, _, _, _ = self.sam_model.sam_mask_decoder(
        #     image_embeddings=image_embeddings["image_embed"],
        #     image_pe=image_embeddings["image_pe"],
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False  # Ensure single output
        # )

        low_res_masks, prd_scores, _, _ = self.sam_mask_decoder(
            image_embeddings=feats,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # Pass through the multiclass head
        return self.multiclass_head(low_res_masks)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def main():
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login()

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


    train_image_path = f'{args.data_path}train/images/'
    train_mask_path = f'{args.data_path}train/masks/'

    test_image_path = f'{args.data_path}test/images/'
    test_mask_path = f'{args.data_path}test/masks/'

    if cfg['resize_method'] == "resize_retain_aspect":
        train_images, train_masks = load_tifs_resize_to_np_retain_ratio(train_image_path, train_mask_path)
        test_images, test_masks = load_tifs_resize_to_np_retain_ratio(test_image_path, test_mask_path)
    elif cfg['resize_method'] == "resize_simple":
        train_images, train_masks = load_tifs_resize_to_np(train_image_path, train_mask_path)
        test_images, test_masks = load_tifs_resize_to_np(test_image_path, test_mask_path)
    else:
        raise NotImplementedError

    train_transforms, test_transforms = None, None


    if args.multiclass:
        cfg['mask_type'] = 'layer'

    # dataset setup
    train_ds = SnowDataset(train_images, train_masks, transforms=train_transforms, mask_type=cfg['mask_type'], size_strategy=cfg['resize_method'], dilate=cfg['dilate'])
    test_ds = SnowDataset(test_images, test_masks, transforms=test_transforms, dilate=cfg['dilate'])

    main_worker(args, train_dataset=train_ds, test_dataset=test_ds, config=cfg)

def main_worker(args, train_dataset, test_dataset, config):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
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

    sam2_model = build_sam2(config_file=model_cfg, ckpt_path=sam2_checkpoint, device=device)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if args.multiclass:
        # assuming sam outputs [batch_size, 256, H, W]
        sam2_model = MulticlassSAMWrapper(sam2_model, args.nclasses) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    predictor = SAM2ImagePredictor(sam2_model)

    # make prompt encoder and mask decoder trainable
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # training setup
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=config['learning_rate'],weight_decay=1e-4) #1e-5, weight_decay = 4e-5
    # mix precision
    scaler = torch.amp.GradScaler(device_str)

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
       class_weights = torch.load('weights_20_2.pt').to(device).to(torch.float32)
       ######## lol
       sparse_embeddings = torch.zeros((1, 1, 256), device=predictor.model.device)
       dense_prompt_embeddings = torch.zeros((1, 256, 256), device=predictor.model.device)
       dense_embeddings = F.interpolate(
           dense_prompt_embeddings.unsqueeze(0),  # Add batch dimension
           size=(64, 64),  # Match expected spatial resolution of src
           mode='bilinear',
           align_corners=False
       ).squeeze(0)  # Remove batch dimension if necessary
       #########
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Initialize scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # 500 , 250, gamma = 0.1
    accumulation_steps = 4



    for epoch in trange(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        with torch.amp.autocast(device_str):
            for batch_idx, tup in enumerate(train_loader):
                image = np.array(tup[0].squeeze(0))
                mask = np.array(tup[1].squeeze(0))

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                if args.multiclass:

                    predictor.set_image(image)



                    # batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                    low_res_masks = predictor.model(sparse_embeddings, dense_embeddings, high_res_features,
                                                    predictor._features["image_embed"][-1].unsqueeze(0))
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])


                    gt_mask = torch.tensor(mask, dtype=torch.long).to(device) - 1 # since starting from 1 and not 0 lol
                    ## we might want to do class weighing
                    loss = F.cross_entropy(prd_masks, gt_mask, weight=class_weights)
                    
                    # IoU computation
                    pred_labels = torch.argmax(prd_masks, dim=1)  # Shape: [batch_size, H, W]
                    # if epoch == 10:
                    #     torch.save(pred_labels, f'{epoch}_pred_mask.pt')
                    #     torch.save(gt_mask, f'{epoch}_gt_mask.pt')
                    iou_per_class = []
                    for cls in range(prd_masks.shape[1]):  # Loop over classes
                        inter = ((pred_labels == cls) & (gt_mask == cls)).sum()
                        union = ((pred_labels == cls) | (gt_mask == cls)).sum()
                        if union > 0:
                            iou_per_class.append((inter / union).item())
                    mean_iou = np.mean(iou_per_class) if iou_per_class else 0

                    if args.use_wandb:
                        for cls, iou in enumerate(iou_per_class):
                            wandb.log({f"iou_class_{cls}": iou})
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ binary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                else:
                    num_masks = tup[3].squeeze(0)

                    if image is None or mask is None or num_masks == 0:
                        print("Continuing because empty image, mask, or no number of masks", flush=True)
                        continue

                    input_prompt = np.array(tup[2].squeeze(0))

                    input_label = np.ones((num_masks, 1))

                    if not isinstance(input_prompt, np.ndarray) or not isinstance(input_label, np.ndarray):
                        print("Continuing because prompt or label is not a numpy array", flush=True)
                        continue

                    if input_prompt.size == 0 or input_label.size == 0:
                        print("Continuing because size of prompt of label is zero", flush=True)
                        continue

                    predictor.set_image(image)
                    _, unnorm_coords, labels, _ = predictor._prep_prompts(input_prompt, input_label, box=None, mask_logits=None, normalize_coords=True)
                    if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                        print("Continuing because of miscellaneous", flush=True)
                        continue

                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels), boxes=None, masks=None,
                    )

                    batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                    gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ binary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


            #    # Apply gradient accumulation
            #     loss = loss / accumulation_steps
            #     scaler.scale(loss).backward()

            #     # Clip gradients
            #     torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            #     if epoch % accumulation_steps == 0:
            #         scaler.step(optimizer)
            #         scaler.update()
            #         predictor.model.zero_grad()

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation logic
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Gradient clipping (optional)
                    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

                    # Step optimizer and scaler
                    scaler.step(optimizer)
                    scaler.update()

                    # Reset gradients
                    optimizer.zero_grad()


                # if epoch == 1:
                #     mean_iou = 0
                    
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ binary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                if not args.multiclass:
                    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ binary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

                print("Epoch " + str(epoch) + ":\t", "Train Accuracy (IoU) = ", mean_iou)
                if args.use_wandb:
                    wandb.log({"epoch": epoch, "train_loss": loss, "train_iou": mean_iou})
            # Update scheduler
            scheduler.step()
                

        # Validate
        for _, tup in enumerate(val_loader):
            image = np.array(tup[0].squeeze(0))
            mask = np.array(tup[1].squeeze(0))

            with torch.no_grad():
                predictor.set_image(image)

                if args.multiclass:
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

                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

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

                    if args.use_wandb:
                        for cls, iou in enumerate(iou_per_class):
                            wandb.log({f"val_iou_class_{cls}": iou})
                        wandb.log({"val_mean_iou": mean_iou})

                    print(f"Validation Mean IoU: {mean_iou}")
                else:
                    input_prompt = np.array(tup[2].squeeze(0))
                    point_labels = np.ones([input_prompt.shape[0], 1])
                    # Binary segmentation validation
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels), boxes=None, masks=None,
                    )

                    batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                    gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    val_iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)

                    val_mean_iou = val_mean_iou * 0.99 + 0.01 * np.mean(val_iou.cpu().detach().numpy()) if epoch > 1 else np.mean(val_iou.cpu().detach().numpy())
                    if args.use_wandb:
                        wandb.log({"val_iou": val_mean_iou})

                    print(f"Validation IoU: {val_mean_iou}")


    FINETUNED_MODEL = FINETUNED_MODEL_NAME + "_" + str(pref) + "_" + str(epoch) + ".torch"
    torch.save(predictor.model.state_dict(), os.path.join("snowpack/model/model_checkpoints", FINETUNED_MODEL))

if __name__ == "__main__":
    main()