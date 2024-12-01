import wandb
import os
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed
from models.unet_torch.encoders import get_preprocessing_fn
from .run_smp_finetune import train as unet_torch_train
from .run_smp_finetune import validate as unet_torch_validate
import models.unet_torch as smp
import argparse
from .utils.preprocess_helpers import get_preprocessing

parser = argparse.ArgumentParser(description="K-fold cross-validation")
parser.add_argument(
    "--final_sweep", default=False, action="store_true"
)
parser.add_argument(
    "--seed", type=int, default=84
)
parser.add_argument(
    "--config", type=str, default="configs/default.json"
)
parser.add_argument(
    "--wandb_entity", type=str, default="sea-ice"
)
parser.add_argument(
    "--num_folds", type=int, default=5
)

sweep_configuration = {
    "name": "sweep_snowpack_hyperparameters",
    "method": "random",
    "metric": {"goal": "maximize", "name": "iou"},
    "parameters": {
        "im_size": {"values": [1024]},
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 1e-2]},
        "batch_size": {"values": [1, 2, 4]},
        "dropout": {"values": [0.0, 0.2, 0.5]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
    },
}

final_sweep_configuration = {
    "name": "sweep_snowpack_hyperparameters",
    "method": "random",
    "metric": {"goal": "maximize", "name": "iou"},
    "parameters": {
        "im_size": {"values": [1024]},
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 1e-2]},
        "batch_size": {"values": [1, 2, 4]},
        "dropout": {"values": [0.0, 0.2, 0.5]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
    },
}

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train_smp_torch(num, args, sweep_id, sweep_run_name, config, train_loader, test_loader, class_weights):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    with open(args.config) as f:
        hyper_config = json.load(f)

    cfg_model = hyper_config['model']
    cfg_training = hyper_config['training']

    cfg_training['dropout'] = config['dropout']

    class_weights_np = class_weights
    class_weights = torch.from_numpy(class_weights).float().cuda(0)

    # create model
    model = smp.create_model(
        arch=args.arch,
        encoder_name=cfg_model["backbone"],
        encoder_weights=cfg_model["pretrain"],
        in_channels=3,
        classes=cfg_model["num_classes"],
        dropout_rate=cfg_training["dropout"],
    )

    torch.cuda.set_device(0)
    model = model.cuda(0)

    # freeze weights in the image_encoder
    if not cfg_model["encoder_freeze"]:
        for name, param in model.named_parameters():
            if param.requires_grad and "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    else:
        for name, param in model.named_parameters():
            if param.requires_grad and "image_encoder" in name or "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    elif config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay'],
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    for epoch in range(150):
        unet_torch_train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            class_weights_np=class_weights_np,
        )
        _, val_miou, val_mp_iou, val_oc_iou, val_si_iou = unet_torch_validate(test_loader, model, epoch, scheduler, cfg_model)

    run.log(dict(val_mean_iou=val_miou, val_melt_pond_iou=val_mp_iou, val_ocean_iou=val_oc_iou, val_sea_ice_iou=val_si_iou))
    run.finish()
    return val_miou, val_mp_iou, val_oc_iou, val_si_iou


def cross_validate():
    args=parser.parse_args()

    with open(args.config) as f:
        hyper_config = json.load(f)

    cfg_model = hyper_config['model']
    cfg_training = hyper_config['training']
    cfg_model['im_size'] = sweep_run.config.im_size

    class_weights = compute_class_weights(y_path)

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    metrics_miou = []
    metrics_mp_iou = []
    metrics_oc_iou = []
    metrics_si_iou = []

    X_path = "./data/images/"
    y_path = "./data/masks/"

    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=14)

    X, y = np.load(X_path), np.load(y_path)

    for num, (train, test) in enumerate(kfold.split(X, y)):
        train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[train], masks=y[train])
        test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[test], masks=y[test])

        train_loader = DataLoader(
            train_dataset,
            batch_size=sweep_run.config.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )
        reset_wandb_env()
        val_miou, val_mp_iou, val_oc_iou, val_si_iou = train_smp_torch(
            sweep_id=sweep_id,
            num=num,
            args=parser.parse_args(),
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
            train_loader=train_loader,
            test_loader=test_loader,
            class_weights=class_weights,
        )
        metrics_miou.append(val_miou)
        metrics_mp_iou.append(val_mp_iou)
        metrics_oc_iou.append(val_oc_iou)
        metrics_si_iou.append(val_si_iou)

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_melt_pond_iou=sum(metrics_mp_iou) / len(metrics_mp_iou)))
    sweep_run.log(dict(val_mean_iou=sum(metrics_miou) / len(metrics_miou)))
    sweep_run.log(dict(val_ocean_iou=sum(metrics_oc_iou) / len(metrics_oc_iou)))
    sweep_run.log(dict(val_sea_ice_iou=sum(metrics_si_iou) / len(metrics_si_iou)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


def main():    
    wandb.login()
    args=parser.parse_args()

    set_seed(args.seed)

    count = 1

    if args.sweep_hyperparameters:
        sweep_config = sweep_configuration
        count = 100
    else:
        sweep_config = final_sweep_configuration

    sweep_id = wandb.sweep(sweep=sweep_config, project="snowpack", entity=args.wandb_entity)
    wandb.agent(sweep_id, function=cross_validate, count=count)

    wandb.finish()

if __name__ == "__main__":
    main()