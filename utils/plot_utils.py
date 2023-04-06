import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pdb
import cv2



def calculate_indices_avg(tensor, index):
    img = tensor.cpu().numpy()
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # pdb.set_trace()
    # blue, green, red = img[..., 0], img[..., 1], img[..., 2]
    blue, green, red = cv2.split(img)
    
    g = np.mean(green)
    r = np.mean(red)
    b = np.mean(blue)
    # print(g, r, b)

    
    if index == "RGBVI":
        RGBVI = (g**2 - (b*r)) / (g**2 + (b*r))
        return RGBVI
    elif index == "GLI":
        GLI = g/r #(2*g - r-b) / (- r - b)
        return GLI
    elif index == "VARI":
        VARI = (g - r) / (g + r - b)
        return VARI
    elif index == "NGRDI":
        NGRDI = (g - r) / (g + r)
        return NGRDI
    else:
        print("Index not found")





RGBVI= "Red-Green-Blue Vegetation Index"
GLI= "Grean Leaf Index"
VARI="Visible Atmospherically Resistant Index"
NGRDI="Normalized Green Red Difference Index"










class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    img_token[mask.detach().cpu()!=0] = mask_value
    img = rearrange(
        img_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def plot_semseg_gt(input_dict, ax=None, image_size=224):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(input_dict['UAV_RGB'].detach().cpu())[0].permute(1,2,0)
    semseg = F.interpolate(
        input_dict['semseg'].unsqueeze(0).cpu().float(), size=image_size, mode='nearest'
    ).long()[0,0]
    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()


def plot_semseg_gt_masked(input_dict, mask, ax=None, mask_value=1.0, image_size=224):
    img = plot_semseg_gt(input_dict, image_size=image_size)
    img = torch.LongTensor(img).permute(2,0,1).unsqueeze(0)
    masked_img = get_masked_image(img.float()/255.0, mask, image_size=image_size, patch_size=16, mask_value=mask_value)
    masked_img = masked_img[0].permute(1,2,0)
    
    if ax is not None:
        ax.imshow(masked_img)
    else:
        return masked_img


def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token[mask.detach().cpu()==0] = gt_token[mask.detach().cpu()==0]
    img = rearrange(
        pred_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img


def plot_predictions(input_dict, preds, masks, id, out_path, image_size=224):

    masked_Sat = get_masked_image(
        denormalize(input_dict['Sat_RGB']), 
        masks['Sat_RGB'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()
    masked_rgb = get_masked_image(
        denormalize(input_dict['UAV_RGB']), 
        masks['UAV_RGB'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()


    pred_Sat = denormalize(preds['Sat_RGB'])[0].permute(1,2,0).clamp(0,1)
    pred_rgb = denormalize(preds['UAV_RGB'])[0].permute(1,2,0).clamp(0,1)

    pred_Sat2 = get_pred_with_input(
        denormalize(input_dict['Sat_RGB']), 
        denormalize(preds['Sat_RGB']).clamp(0,1), 
        masks['Sat_RGB'],
        image_size=image_size
    )[0].permute(1,2,0).detach().cpu()
    pred_rgb2 = get_pred_with_input(
        denormalize(input_dict['UAV_RGB']), 
        denormalize(preds['UAV_RGB']).clamp(0,1), 
        masks['UAV_RGB'],
        image_size=image_size
    )[0].permute(1,2,0).detach().cpu()

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0)

    grid[0].imshow(masked_Sat)
    grid[1].imshow(pred_Sat)
    grid[2].imshow(denormalize(input_dict['Sat_RGB'])[0].permute(1,2,0).detach().cpu())

    grid[3].imshow(masked_rgb)
    grid[4].imshow(pred_rgb)
    grid[5].imshow(denormalize(input_dict['UAV_RGB'])[0].permute(1,2,0).detach().cpu())

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
                
    fontsize = 16
    grid[0].set_title('Masked inputs', fontsize=fontsize)
    grid[1].set_title('MultiMAE predictions', fontsize=fontsize)
    grid[2].set_title('Original Reference', fontsize=fontsize)
    grid[0].set_ylabel('Sat_RGB', fontsize=fontsize)
    grid[3].set_ylabel('UAV_RGB', fontsize=fontsize)
    plt.savefig(out_path + 'epoch'+str(id)+'.png')
    return {
        'Sat_input': masked_Sat,
        'Sat_pred': pred_Sat2,
        'Sat_gt': denormalize(input_dict['Sat_RGB'])[0].permute(1,2,0).detach().cpu(),
        'rgb_input': masked_rgb,
        'rgb_pred': pred_rgb2,
        'rgb_gt': denormalize(input_dict['UAV_RGB'])[0].permute(1,2,0).detach().cpu(),
    }










def plot_predictions_test(input_dict, preds, masks, id, out_path,sattest,uavtest, mask__, image_size=224):
    # pdb.set_trace()
    pred_Sat = denormalize(preds['Sat_RGB'])[0].permute(1,2,0).clamp(0,1) #* TF.to_tensor(mask__).T
    pred_rgb = denormalize(preds['UAV_RGB'])[0].permute(1,2,0).clamp(0,1)#* TF.to_tensor(mask__).T

    sat_in = denormalize(input_dict['Sat_RGB'])[0].permute(1,2,0).detach().cpu()#* TF.to_tensor(mask__).T
    uav_in = denormalize(input_dict['UAV_RGB'])[0].permute(1,2,0).detach().cpu()#* TF.to_tensor(mask__).T
    # pdb.set_trace()




    sat_1_RGBVI_avg = calculate_indices_avg(sat_in, "RGBVI")
    sat_1_GLI_avg = calculate_indices_avg(sat_in, "GLI")
    sat_1_VARI_avg = calculate_indices_avg(sat_in, "VARI")
    sat_1_NGRDI_avg = calculate_indices_avg(sat_in, "NGRDI")

    sat_2_RGBVI_avg = calculate_indices_avg(pred_Sat, "RGBVI")
    sat_2_GLI_avg = calculate_indices_avg(pred_Sat, "GLI")
    sat_2_VARI_avg = calculate_indices_avg(pred_Sat, "VARI")
    sat_2_NGRDI_avg = calculate_indices_avg(pred_Sat, "NGRDI")



    uav_1_RGBVI_avg = calculate_indices_avg(uav_in, "RGBVI")
    uav_1_GLI_avg = calculate_indices_avg(uav_in, "GLI")
    uav_1_VARI_avg = calculate_indices_avg(uav_in, "VARI")
    uav_1_NGRDI_avg = calculate_indices_avg(uav_in, "NGRDI")

    uav_2_RGBVI_avg = calculate_indices_avg(pred_rgb, "RGBVI")
    uav_2_GLI_avg = calculate_indices_avg(pred_rgb, "GLI")
    uav_2_VARI_avg = calculate_indices_avg(pred_rgb, "VARI")
    uav_2_NGRDI_avg = calculate_indices_avg(pred_rgb, "NGRDI")


    sattest = sattest.split("/")[-1][:-4]+sattest.split("/")[-2]+".jpg"
    uavtest = uavtest.split("/")[-1][:-4]+uavtest.split("/")[-2]+".jpg"
    # pdb.set_trace()

    torchvision.utils.save_image(pred_Sat.T, "/work/mech-ai/ayanlade/sat_uav/su_multimae/output/gen/sat/"+sattest)
    torchvision.utils.save_image(pred_rgb.T, "/work/mech-ai/ayanlade/sat_uav/su_multimae/output/gen/uav/"+uavtest)
    torchvision.utils.save_image(sat_in.T, "/work/mech-ai/ayanlade/sat_uav/su_multimae/output/in/sat/"+sattest)
    torchvision.utils.save_image(uav_in.T, "/work/mech-ai/ayanlade/sat_uav/su_multimae/output/in/uav/"+uavtest)
    
    
    return  sat_1_RGBVI_avg, sat_1_GLI_avg, sat_1_VARI_avg,sat_1_NGRDI_avg, sat_2_RGBVI_avg, sat_2_GLI_avg, sat_2_VARI_avg,sat_2_NGRDI_avg, uav_1_RGBVI_avg,    uav_1_GLI_avg , uav_1_VARI_avg ,    uav_1_NGRDI_avg ,    uav_2_RGBVI_avg , uav_2_GLI_avg ,    uav_2_VARI_avg , uav_2_NGRDI_avg 