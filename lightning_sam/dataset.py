import os
import random
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# from pycocotools.coco import COCO
from mobile_sam.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(
            self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if self.transform:
            image, masks, bboxes = self.transform(
                image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()


class MaskSegmentDataset(Dataset):
    def __init__(self, list_image_file, path_occlusion_object, transform, root_dir=None) -> None:
        super().__init__()
        self.list_name_img = np.load(list_image_file)
        self.root_dir = root_dir

        self.path_occlusion_object = path_occlusion_object
        self.list_path_occlusion_object = os.listdir(path_occlusion_object)
        self.transform = transform

    def mask_random(self, image, occlusion_object=None, ratio_height=-1):
        if ratio_height is None:
            ratio_height = np.clip(np.random.rand(), 0.3, 0.65)

        # if ratio_width is None:
        #     ratio_width = np.clip(np.random.rand(), 0.1, 0.5)
        ratio_width = ratio_height

        image = np.copy(image)

        height, width = image.shape[0], image.shape[1]

        if ratio_height == -1:
            occ_height, occ_width = image.shape[:2]
            occ_height = min(height, occ_height)
            occ_width = min(width, occ_width)
        else:
            occ_height, occ_width = int(
                height * ratio_height), int(width * ratio_width)

        row_start = np.random.randint(0, height - int(height * ratio_height))
        row_end = min(row_start + occ_height, height)

        col_start = np.random.randint(0, width - int(width * ratio_width))
        col_end = min(col_start + occ_width, width)

        occ_width = col_end - col_start
        occ_height = row_end - row_start

        if occlusion_object is not None:

            occlusion_object = cv2.resize(
                occlusion_object, (occ_width, occ_height))
            occlu_image, mask = occlusion_object[:,
                                                 :, :3], occlusion_object[:, :, 3:]
            occlu_image = occlu_image[:, :, ::-1]

            image[row_start:row_end, col_start:col_end, :] = occlu_image * \
                mask + image[row_start:row_end,
                             col_start:col_end, :] * (1 - mask)
        else:
            occlusion_noise = np.random.rand(occ_height, occ_width, 3)
            occlusion_noise = np.array(occlusion_noise * 255, dtype=np.uint8)
            image[row_start:row_end, col_start:col_end, :] = occlusion_noise

        return image

    def augment_occlusion(self, image):

        # for 4 channels npy
        mask_image = np.load(os.path.join(
            self.path_occlusion_object, random.choice(self.list_path_occlusion_object)))

        # for image
        # mask_image = cv2.imread(os.path.join(self.path_occlusion_object, random.choice(self.list_path_occlusion_object)))
        # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        image_augment = self.mask_random(image, mask_image, ratio_height=None)
        mask = np.where(np.array(image) != np.array(image_augment), 1, 0)
        mask = mask

        image_augment = np.array(image_augment)
        mask = np.array(mask)

        return image_augment, mask

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(
            self.root_dir, self.list_name_img[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))

        image_augment, masks = self.augment_occlusion(image)
        image_augment = np.array(image_augment, np.uint8)
        masks = [np.array(masks[:, :, 0] * 1.0, dtype=np.float32)]
        if self.transform:
            image_augment, masks = self.transform(image_augment, masks)
        masks = np.stack(masks, axis=0)

        return image_augment, torch.tensor(masks).float()

    def __len__(self):
        return len(self.list_name_img)

class EvalDataset(Dataset):
    def __init__(self, image_dir) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.list_image = os.listdir(image_dir) 

    
    def __getitem__(self, index: Any) -> Any:
        image = cv2.imread(os.path.join(
            self.image_dir, self.list_image[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))

        return image, None
    
    def __len__(self):
        return len(self.list_image)
    
def collate_fn(batch):
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    return images, bboxes


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask))
                 for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        if bboxes is not None:
            # Adjust bounding boxes
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] +
                       pad_w, bbox[3] + pad_h] for bbox in bboxes]

            return image, masks, bboxes
        else:
            return image, masks


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader


def load_custom_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = MaskSegmentDataset(
        list_image_file='/home1/data/tanminh/NML-Face/list_name_file/list_name_train_no_masked.npy',
        root_dir="/home1/data/FFHQ/StyleGAN_data256_jpg",
        path_occlusion_object="/home1/data/tanminh/Face_Deocclusion_Predict_Masked/images/occlusion_object/clean_segment",
        transform=transform
    )
    val = MaskSegmentDataset(
        list_image_file='/home1/data/tanminh/NML-Face/list_name_file/list_name_val_no_masked.npy',
        root_dir="/home1/data/FFHQ/StyleGAN_data256_jpg",
        path_occlusion_object="/home1/data/tanminh/Face_Deocclusion_Predict_Masked/images/occlusion_object/clean_segment",
        transform=transform
    )

    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers)
    return train_dataloader, val_dataloader
