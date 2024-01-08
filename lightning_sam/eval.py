import os 

import cv2
import torch 
import numpy as np 
from tqdm import tqdm 
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

from model import Model 

from dataset import MaskSegmentDataset, EvalDataset

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def visualize_custom(cfg): 
    model = Model(cfg)
    model.setup()
    model.eval()
    model.to("cpu")
    dataset = EvalDataset(
        image_dir= "/home1/data/tanminh/Face_Deocclusion_Predict_Masked/occlu_face_dir"
    )
    device= "cpu"
    points = torch.from_numpy(np.array([[30, 52], [65, 52], [34, 92], [63, 92]])) * 1024 / 112 
    label = torch.from_numpy(np.array([1,1,1,1]))
    points = points.to(device)
    label = label.to(device)

    predictor = model.get_predictor()
    out_dir = "result_image"
    os.makedirs(out_dir, exist_ok=True)

    for id in tqdm(range(len(dataset))): 
        image, ori_masks = dataset[id] 
        input_tensor = torch.from_numpy(np.array(image/ 255.0, np.float32) )
        input_tensor = torch.permute(input_tensor, (2, 0, 1))
        input_tensor = torch.unsqueeze(input_tensor, 0)
        input_tensor = input_tensor.to(device)
        print("input tensor shape: ", input_tensor.shape)
        pred_mask, _ = model(input_tensor, (points[None, :, :], label[None, :])) 

        image_output_path = os.path.join(out_dir, str(id) + ".jpg")
        pred_mask = (pred_mask[0])
        pred_mask = pred_mask > 0.5 
        # ori_masks = torch.from_numpy(np.array(ori_masks, dtype= np.bool_))
        # image_output = draw_image(image, (ori_masks[0]), boxes=None, labels=None)
        image_output = draw_image(image, (pred_mask[0]), boxes=None, labels=None)
        cv2.imwrite(image_output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))


from config import cfg 
visualize_custom(cfg)
