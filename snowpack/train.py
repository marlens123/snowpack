import torch
import wandb
import time
import numpy as np


def train(
    train_loader,
    class_weights,
    model,
    optimizer,
    epoch,
    cfg_model,
    args=None,
    class_weights_np=None,
):

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    dice_loss = SoftDiceLoss(
        batch_dice=True, do_bg=False, rebalance_weights=class_weights_np
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    be_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):

        if gpu != "cpu":
            img = tup[0].float().cuda(gpu, non_blocking=True)
            label = tup[1].long().cuda(gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()
        b, _, h, w = img.shape

        logits_mask = model.forward(img)

        if cfg_model["task"] == "multiclass":
            assert label.max() <= 16 and label.min() >= 0
            pred_softmax = F.softmax(logits_mask, dim=1)
            loss = dice_loss(pred_softmax, label.squeeze(1)) + ce_loss(logits_mask, label.squeeze(1))
        elif cfg_model["task"] == "binary":
            assert label.max() <= 1 and label.min() >= 0
            loss = be_loss(logits_mask, label.squeeze(1))
        else:
            raise ValueError("Task not recognized.")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if use_wandb:
        wandb.log({"epoch": epoch, "train_loss": loss})


def validate(val_loader, model, epoch, scheduler, cfg_model, args=None, writer=None):
    loss_list = []
    jac_mean = []
    output_list = []
    multiclass_dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    binary_dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    model.eval()

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            if gpu != "cpu":
                img = tup[0].float().cuda(gpu, non_blocking=True)
                label = tup[1].long().cuda(gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, _, h, w = img.shape

            # compute output
            mask = model.forward(img)

            if cfg_model["task"] == "multiclass":
                assert label.max() <= 16 and label.min() >= 0
                pred_softmax = F.softmax(mask, dim=1)
                loss = multiclass_dice_loss(pred_softmax, label.squeeze(1))

                jaccard = JaccardIndex(
                    task="multiclass", num_classes=cfg_model["num_classes"], average="macro"
                ).to(gpu)
                jac_m = jaccard(pred_softmax, label.squeeze(1))
            elif cfg_model["task"] == "binary":
                assert label.max() <= 1 and label.min() >= 0
                loss = binary_dice_loss(mask, label)
                prob_mask = mask.sigmoid()[0]
                pred_mask = (prob_mask > 0.5).float()
                assert pred_mask.shape == (1, 1024, 1024)
        
                jaccard = JaccardIndex(
                    task="binary"
                ).to(gpu)
                jac_m = jaccard(prob_mask, label.squeeze(1))
            else:
                raise ValueError("Task not recognized.")
            loss_list.append(loss.item())

            jac_mean.append(jac_m.item())

            if use_wandb:
                wandb.log({"epoch": epoch, "val_loss_{}".format(i): loss.item()})
                wandb.log({"epoch": epoch, "val_jac_{}".format(i): jac_m.item()})

    if use_wandb:
        wandb.log({"epoch": epoch, "val_loss": np.mean(loss_list)})
        wandb.log({"epoch": epoch, "val_jac": np.mean(jac_mean)})

    if epoch >= 10:
        scheduler.step(np.mean(loss_list))

    return np.mean(loss_list), np.mean(jac_mean)