import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn 
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def multiclass_epoch(train_loader, predictor, accumulation_steps, epoch, optimizer):

    for param in predictor.model.parameters(): param.data = param.data.half()

    epoch_mean_iou, loss_mean_iou = [], []
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
    for batch_idx, tup in enumerate(train_loader):
        image = np.array(tup[0].squeeze(0))
        mask = np.array(tup[1].squeeze(0))

        predictor.model.float()

        dtype = torch.float32
        sparse_embeddings = sparse_embeddings.to(dtype)
        dense_embeddings = dense_embeddings.to(dtype)

        predictor.set_image(image)

        # batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

        low_res_masks = predictor.model(sparse_embeddings, dense_embeddings, high_res_features,
                                        predictor._features["image_embed"][-1].unsqueeze(0))
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]).float()

        gt_mask = torch.tensor(mask, dtype=torch.long).to(prd_masks.device)

        assert gt_mask.min() >= 0, "gt_mask contains negative indices"
        assert gt_mask.max() < 21, "gt_mask contains out-of-range indices"
        print(f"Prediction shape: {prd_masks.shape}, Ground truth shape: {gt_mask.shape}")
        ## we might want to do class weighing
        #loss = F.cross_entropy(prd_masks, gt_mask, weight=class_weights.to(prd_masks.dtype).to(prd_masks.device))
        loss = F.cross_entropy(prd_masks, gt_mask)

        # IoU computation
        pred_labels = torch.argmax(prd_masks, dim=1)  # Shape: [batch_size, H, W]

        iou_per_class = []
        for cls in range(prd_masks.shape[1]):  # Loop over classes
            inter = ((pred_labels == cls) & (gt_mask == cls)).sum()
            union = ((pred_labels == cls) | (gt_mask == cls)).sum()
            if union > 0:
                iou_per_class.append((inter / union).item())
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0

        loss = loss / accumulation_steps  # Scale the loss for accumulation
        loss.backward()

        # Gradient accumulation logic
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Clip gradients after scaling the gradients and before stepping the optimizer
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            # Step the optimizer
            optimizer.step()
            
            # Reset gradients
            optimizer.zero_grad()

        epoch_mean_iou.append(mean_iou)
        loss_mean_iou.append(loss.item())
        print("Epoch " + str(epoch) + ":\t", "Train Accuracy (IoU) = ", mean_iou, f' Loss = {loss.detach().item()}')
    return np.mean(epoch_mean_iou), np.mean(loss_mean_iou)


                
def validate_multiclass(val_loader, predictor, device):
    epoch_mean_iou = []
    for _, tup in enumerate(val_loader):
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
    return np.mean(epoch_mean_iou)


def binary_epoch(train_loader, predictor, accumulation_steps, epoch, optimizer, device):

    for param in predictor.model.parameters(): param.data = param.data.half()

    sparse_embeddings = torch.zeros((1, 1, 256), device=predictor.model.device)
    dense_prompt_embeddings = torch.zeros((1, 256, 256), device=predictor.model.device)
    dense_embeddings = F.interpolate(
        dense_prompt_embeddings.unsqueeze(0),  # Add batch dimension
        size=(64, 64),  # Match expected spatial resolution of src
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension if necessary


    epoch_mean_iou, loss_mean_iou = [], []

    for _, tup in enumerate(train_loader):
        image = np.array(tup[0].squeeze(0))
        mask = np.array(tup[1].squeeze(0))
        #input_prompt = np.array(tup[2].squeeze(0))
        #num_masks = tup[3].squeeze(0)

        predictor.model.float()

        #if image is None or mask is None or num_masks == 0:
        #    print("Continuing because empty image, mask, or no number of masks", flush=True)
        #    continue

        #input_label = np.ones((num_masks, 1))

        #if not isinstance(input_prompt, np.ndarray) or not isinstance(input_label, np.ndarray):
        #    print("Continuing because prompt or label is not a numpy array", flush=True)
        #    continue

        #if input_prompt.size == 0 or input_label.size == 0:
        #    print("Continuing because size of prompt of label is zero", flush=True)
        #    continue

        predictor.set_image(image)
        #_, unnorm_coords, labels, _ = predictor._prep_prompts(input_prompt, input_label, box=None, mask_logits=None, normalize_coords=True)
        #if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
        #    print("Continuing because of miscellaneous", flush=True)
        #    continue

        #sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        #    points=(unnorm_coords, labels), boxes=None, masks=None,
        #)

        dtype = torch.float32
        sparse_embeddings = sparse_embeddings.to(dtype)
        dense_embeddings = dense_embeddings.to(dtype)

        #batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]).float()

        gt_mask = torch.tensor(mask, dtype=torch.long).to(prd_masks.device)
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps

        for param in predictor.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                    print(f"Invalid gradient detected for parameter {param}")
                    break

        loss.backward()

        # Clip gradients  
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

        if epoch % accumulation_steps == 0:
            # Step the optimizer
            optimizer.step()
            predictor.model.zero_grad()

        epoch_mean_iou.append(iou.cpu())
        loss_mean_iou.append(loss.item())

        print("Epoch " + str(epoch) + ":\t", "Train Accuracy (IoU) = ", iou)
 
    return np.mean(epoch_mean_iou), np.mean(loss_mean_iou)


def validate_binary(val_loader, predictor, device, val_mean_iou):

    epoch_mean_iou = []
    for _, tup in enumerate(val_loader):
        image = np.array(tup[0].squeeze(0))
        mask = np.array(tup[1].squeeze(0))

        with torch.no_grad():
            predictor.set_image(image)

            sparse_embeddings = torch.zeros((1, 1, 256), device=predictor.model.device)
            dense_prompt_embeddings = torch.zeros((1, 256, 256), device=predictor.model.device)
            dense_embeddings = F.interpolate(
                dense_prompt_embeddings.unsqueeze(0),  # Add batch dimension
                size=(64, 64),  # Match expected spatial resolution of src
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension if necessary

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
            prd_mask = torch.sigmoid(prd_masks[:, 0])

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
    return np.mean(epoch_mean_iou)


class MulticlassSAMWrapper(nn.Module):
    def __init__(self, sam_model, n_classes):
        super(MulticlassSAMWrapper, self).__init__()
        self.model = sam_model
        self.sam_mask_decoder = self.model.sam_mask_decoder
        self.sam_prompt_encoder = self.model.sam_prompt_encoder
        self.image_size = self.model.image_size #### we can probably change this
        self.no_mem_embed = self.model.no_mem_embed


        self.device = self.model.device


        self.multiclass_head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
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