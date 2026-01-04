#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import torch
import imageio
import numpy as np
import argparse
from PIL import Image
from medpy.metric import binary
from tools.evaluate_utils import dice_coeff

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", default="./experiments/nodule", type=str)
parser.add_argument("--cam_dir", default="cam", type=str)
parser.add_argument('--cam_method', default='normal', type=str, help="[normal, msf]")
parser.add_argument('--save_png', default='cam_png', type=str)
parser.add_argument('--gt_dir', default='./data2/label', type=str)
parser.add_argument("--num_classes", default=2, type=int)  # 1
parser.add_argument("--weights", default="alpha_10_parr_base_ss_sc_cs_cc320_metric2_cls_resnet50_ConsCAM_lr0.001_bs4_epoch100_unfixed.pth", type=str)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    args = parser.parse_args()
    cam_dir = os.path.join(args.work_dir, args.cam_dir, args.weights[0:-4], args.cam_method)
    cam_list = os.listdir(cam_dir)
    
    thresholds = list(np.arange(0.0, 1.0, 0.05))
    best_thr = 0.
    best_dice = 0.
    best_hd95 = float('inf')

    total_dice_per_th = {th: 0.0 for th in thresholds}
    total_hd95_per_th = {th: 0.0 for th in thresholds}
    total_images = len(cam_list)  

    for threshold in thresholds:
        n_cl = args.num_classes
        hist = np.zeros((n_cl, n_cl))
        for cam_id in cam_list:
            # cam_dict = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True).item()
            # cam = cam_dict[0]
            cam = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True)

            cam = np.pad(cam[np.newaxis, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            cam_png = np.argmax(cam, axis=0)
            # compute hist
            cam_png[cam_png > 0] = 1
            # label_id = cam_id.replace('Image', 'Label')[0:-4]
            label_id = cam_id[0:-4]
            # label_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.' + cam_id[0:-4]
            gt = Image.open(os.path.join(args.gt_dir, label_id))
            gt = np.array(gt)
            gt[gt > 0] = 1
            # gt[gt == 5] = 1
            # gt[gt > 1] = 0
            hh, ww = np.shape(gt)
            hist += fast_hist(gt.flatten(), cam_png.flatten(), n_cl)

            input_tensor = torch.from_numpy(cam_png).unsqueeze(0).unsqueeze(0).float() 
            target_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).long()  
            dice_score_func = dice_coeff(input_tensor, target_tensor, reduce_batch_first=False)

            # HD95 (Hausdorff Distance at 95th percentile)
            pred_mask_bin = (cam_png > 0).astype(np.uint8)
            gt_mask_bin = (gt > 0).astype(np.uint8)
            hd95 = binary.hd95(pred_mask_bin, gt_mask_bin)
 
            # Dice & HD95
            total_dice_per_th[threshold] += dice_score_func
            total_hd95_per_th[threshold] += hd95

        # per-class IoU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        # per-class Dice
        dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
          
        if dice[1] > best_dice:
            best_thr = threshold
            best_dice = dice[1]
        print('Threshold:%.2f' % threshold,
              'IoU:%.2f' % (iou[1] * 100),
              'Dice:%.2f' % (dice[1] * 100))
        
    print('Best Threshold:%.2f' % best_thr,
          'Best IoU:%.2f' % (100*best_dice/(2-best_dice)),
          'Best Dice:%.2f' % (best_dice*100))
    
    avg_dice_per_th = {th: total_dice_per_th[th] / total_images for th in thresholds}
    avg_hd95_per_th = {th: total_hd95_per_th[th] / total_images for th in thresholds}

    print("===== Performance metrics for all thresholds =====")
    for th in thresholds:
        print(f'Threshold: {th:.2f} | Avg Dice: {avg_dice_per_th[th]:.4f} | Avg HD95: {avg_hd95_per_th[th]:.2f}')

    best_th_dice = max(avg_dice_per_th, key=avg_dice_per_th.get)
    best_th_hd95 = min(avg_hd95_per_th, key=avg_hd95_per_th.get)

    best_dice = avg_dice_per_th[best_th_dice]
    best_hd95 = avg_hd95_per_th[best_th_hd95]

    print(f'Best Threshold (Dice): {best_th_dice:.2f} | Best Dice: {best_dice:.4f}')
    print(f'Best Threshold (HD95): {best_th_hd95:.2f} | Best HD95: {best_hd95:.2f}')


    if args.save_png:
        print('Saving cam_png...')
        save_dir = create_directory(os.path.join(args.work_dir, args.save_png, args.weights[0:-4], args.cam_method))
        for cam_id in cam_list:
            # cam_dict = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True).item()
            # cam = cam_dict[0]
            cam = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True)

            cam = np.pad(cam[np.newaxis, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thr)
            cam_png = np.argmax(cam, axis=0)
            imageio.imsave(os.path.join(save_dir, cam_id[0:-4]), (cam_png * 255).astype(np.uint8))
        print('Saved!')
