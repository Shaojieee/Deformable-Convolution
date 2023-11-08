from model import resnet
from data import generate_torch_dataset, fashionmnist_image_transform, cifar10_image_transform
from parse_args import offset_parse_args
import os

import cv2
import numpy as np
import torch 
import torch.nn as nn
from torchvision import transforms as T
from tqdm import tqdm


def scale_transformation(img, scale_factor=1.0, borderValue=0):
    img_h, img_w = img.shape[0:2]
    
    cx = img_w//2
    cy = img_h//2
    
    tx = cx - scale_factor * cx
    ty = cy - scale_factor * cy
            
    # scale matrix
    sm = np.float32([[scale_factor, 0, tx],
                    [0, scale_factor, ty]])  # [1, 0, tx], [1, 0, ty]

    img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=borderValue)
    return img

def rotation_transformation(img, angle=3., borderValue=0):
    img_h, img_w = img.shape[0:2]
    rm = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle=angle, scale=1.0) # rotation matrix
    img = cv2.warpAffine(img, rm, (img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return img

def random_rotation(img, scale_factor=1.0, borderValue=0):
    img_h, img_w = img.shape[0:2]
    
    cx = img_w//2
    cy = img_h//2
    
    tx = cx - scale_factor * cx
    ty = cy - scale_factor * cy
            
    # scale matrix
    sm = np.float32([[scale_factor, 0, tx],
                    [0, scale_factor, ty]])  # [1, 0, tx], [1, 0, ty]

    img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=borderValue)
    return img

def plot_offsets(img, save_output, roi_x, roi_y):
    cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=1, thickness=-1)
    input_img_h, input_img_w = img.shape[:2]
    for offsets in save_output.outputs:
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h/offset_tensor_h, input_img_w/offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0] # remove batch axis
        sampling_x = sampling_x[0] # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]
        
        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


def generate_offsets():
    args = offset_parse_args()

    image_file = args.image_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_weights = args.model_weights

    model = resnet(
        pretrained=True, 
        num_classes=args.num_classes,
        version=args.resnet_version, 
        dcn=args.with_deformable_conv,
        unfreeze_conv=args.unfreeze_conv,
        unfreeze_offset=args.unfreeze_offset,
        unfreeze_fc=args.unfreeze_fc,
    )

    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))


    save_output = SaveOutput()

    for name, layer in model.named_modules():
        if "offset" in name and isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(save_output)


    scale_factors = np.arange(0.5, 1.5, 0.01)
    scale_factors = np.concatenate([scale_factors, scale_factors[::-1]])

    rotation_factors = np.arange(-15, 15, 1)
    rotation_factors = np.concatenate([rotation_factors, rotation_factors[::-1]])

    scale_idx_factor = 0
    rotation_idx_factor = 0


    images = []
    
    fps = args.fps
    duration = args.duration
    with torch.no_grad():
        # 24 fps * 10 sec
        for i in tqdm(range(fps*duration)):
            image = cv2.imread(image_file)
            image = cv2.resize(image, (224,224))
            input_img_h, input_img_w, input_img_c = image.shape

            if args.video:
                image = scale_transformation(image, scale_factor=scale_factors[scale_idx_factor])
                image = rotation_transformation(image, angle=rotation_factors[rotation_idx_factor])
                scale_idx_factor = (scale_idx_factor + 1) % len(scale_factors)
                rotation_idx_factor = (rotation_idx_factor + 1) % len(rotation_factors)
            
            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.view(1, input_img_c, input_img_h, input_img_w)
            image_tensor = image_tensor.type(torch.float)
            image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            model.eval()
            out = model(image_tensor)
            
            roi_y, roi_x = input_img_h//2, input_img_w//2        
            plot_offsets(image, save_output, roi_x=roi_x, roi_y=roi_y)
            
            save_output.clear()
            image = cv2.resize(image, dsize=(224,224))

            if args.video==False:
                cv2.imwrite(f'{output_dir}/offsets.png', image)
                break
            else:
                images.append(image)
    
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(f'{output_dir}/offsets.mp4', fourcc, fps, (224, 224))

        for j in range(fps*duration):
            video.write(images[j])

        video.release()
            
    

if __name__=='__main__':
    generate_offsets()
