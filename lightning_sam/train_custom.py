import os
import time

import torch
import numpy as np 
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_custom_datasets
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou


def validate(model, val_dataloader, step):
    model.eval()
    state_dict = model.model.state_dict()
    torch.save(state_dict, os.path.join(cfg.out_dir, f"step-{step:06d}-ckpt.pth"))
    model.train() 


def train(cfg:Box, model:Model, optimizer, scheduler, train_dataloader, val_dataloader):
    
    device = "cuda"
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    step = 0

    points = torch.from_numpy(np.array([[30, 52], [65, 52], [34, 92], [63, 92]])) * 1024 / 112 
    label = torch.from_numpy(np.array([1,1,1,1]))
    points = points.to(device)
    label = label.to(device)
    model = model.to(device)

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()


        for iter, data in enumerate(train_dataloader):
            if step > 1 and step % cfg.eval_interval == 0:
                # validate(fabric, model, val_dataloader, step)
                validate(model, None, step)
            

            print("[INFO] Training")
            step += 1 
            data_time.update(time.time() - end)
            images, gt_masks = data
            images = images.to(device)

            # gt_masks = gt_masks.to(fabric.device)
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, (points[None, :, :], label[None, :]))
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=device)
            loss_dice = torch.tensor(0., device=device)
            loss_iou = torch.tensor(0., device=device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                gt_mask = gt_mask.to(device)
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)
            print("Done 1")
            print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

def main(cfg:Box) -> None: 
    model= Model(cfg)
    model.setup() 

    train_data, val_data = load_custom_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model)

    train(cfg, model, optimizer, scheduler, train_data, val_data)

if __name__ == "__main__":
    main(cfg) 

