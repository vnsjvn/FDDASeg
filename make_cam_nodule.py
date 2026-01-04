#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib
import argparse
from tools import pyutils, imutils, torchutils, trans_utils, evaluate_utils
from data import lung_nodule_dataset
import torch.nn.functional as F
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)  # 1
parser.add_argument("--num_workers", default=8, type=int)  # 1
parser.add_argument("--train_list", default="data/test_labels.txt", type=str)  # 1
parser.add_argument("--data_root", default='./data2/nodule/', type=str)  # 1

###############################################################################
# Network
###############################################################################
parser.add_argument("--network", default="network.resnet50_ConsCAM", type=str)  # 1
# parser.add_argument("--weights", default="pretrained_model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params", type=str)  # 1
parser.add_argument("--weights", default="alpha_10_parr_base_ss_sc_cs_cc320_metric2_cls_resnet50_ConsCAM_lr0.001_bs4_epoch100_unfixed.pth", type=str)  # 1

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument("--crop_size", default=512, type=int)  # 1
parser.add_argument('--min_image_size', default=320, type=int)  # 1
parser.add_argument('--max_image_size', default=640, type=int)  # 1
parser.add_argument("--batch_size", default=16, type=int)  # 1
parser.add_argument("--max_epoches", default=10, type=int)  # 1
parser.add_argument("--lr", default=0.001, type=float)  # 1
parser.add_argument("--wt_dec", default=5e-4, type=float)  # 1
parser.add_argument('--num_pieces', default=4, type=int)  # 1
parser.add_argument('--alpha1', default=1, type=int, help="weight for s_cons_loss")  # 1
parser.add_argument('--alpha2', default=1, type=int, help="weight for p_cons_loss")  # 1
parser.add_argument('--cam_method', default='normal', type=str, help="[normal, msf]")  # 1     msf（Multi-scale Fusion）

###############################################################################
# Save
###############################################################################
parser.add_argument("--work_dir", default="./experiments/nodule", type=str)  # 1
parser.add_argument('--print_ratio', default=0.01, type=float)  # 1
# parser.add_argument("--tensorboard_dir", default="base_lr0.01_bs16_parr_epoch10_metric2", type=str)  # 1
parser.add_argument("--cam_dir", default="cam", type=str)  # 1


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    print(vars(args))
    torchutils.set_seed(args.seed)
    model_dir = create_directory(f'{args.work_dir}/model/')
    model_path = model_dir + f'{args.weights}'
    # tensorboard_dir = create_directory(f'{args.work_dir}/tensorboards/{args.tensorboard_dir}/')
    cam_dir = create_directory(f'{args.work_dir}/{args.cam_dir}/{args.weights[0:-4]}/{args.cam_method}')
    # cam_dir = model_path

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        imutils.RandomResizeLong(args.min_image_size, args.max_image_size),
        transforms.RandomHorizontalFlip(),
        torchutils.Normalize(mean=data_mean, std=data_std),
        imutils.RandomCrop(args.crop_size),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])
    test_transform = transforms.Compose([
        torchutils.Normalize(data_mean, data_std),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])
    test_transform_msf = transforms.Compose([
        torchutils.Normalize(data_mean, data_std),
        imutils.HWC_to_CHW
    ])
    infer_dataset_normal = lung_nodule_dataset.LungNoduleClsDataset(args.train_list, data_root=args.data_root,
                                                               transform=test_transform)
    infer_dataset_msf = lung_nodule_dataset.LungNoduleClsDatasetMSF(args.train_list, data_root=args.data_root,
                                                               scales=[0.5, 1.0, 1.5, 2.0],
                                                               inter_transform=test_transform_msf)

    ###################################################################################
    # Network
    ###################################################################################
    model = getattr(importlib.import_module(args.network), 'NetWithCAM')()
    # model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model.cuda()
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    model_replicas = torch.nn.parallel.replicate(model, list(range(the_number_of_gpu)))
    print(model)


    #################################################################################################
    # Infer
    #################################################################################################
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


    if args.cam_method == 'normal':
        infer_data_loader = DataLoader(infer_dataset_normal, batch_size=1, num_workers=1, drop_last=False)
        for iter, pack in enumerate(infer_data_loader):
            img_names = pack[0]
            images = pack[1].cuda()
            labels = pack[2].cuda()
            img_path = lung_nodule_dataset.get_img_path(img_names[0], args.data_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]
            # _, features = model(images)
            _, _, _, _, features = model(images)
            mask = labels.unsqueeze(2).unsqueeze(3)
            features = F.upsample(features, orig_img_size, mode='bilinear', align_corners=False)
            cams = (torchutils.make_cam(features) * mask)
            cam = cams[0, 0, :, :].cpu().detach().numpy()
            np.save(os.path.join(cam_dir, img_names[0] + '.npy'), cam)
            print('\r# Inferring [{}/{}] = {:.2f}%'.format(iter + 1, len(infer_data_loader),
                                                           (iter + 1) / len(infer_data_loader) * 100))
    else:
        infer_data_loader = DataLoader(infer_dataset_msf, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for iter, pack in enumerate(infer_data_loader):
            img_names = pack[0]
            images = pack[1]
            labels = pack[2]

            img_path = lung_nodule_dataset.get_img_path(img_names[0], args.data_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]


            def _work(i, img):
                with torch.no_grad():
                    with torch.cuda.device(i % the_number_of_gpu):
                        _, cam = model_replicas[i % the_number_of_gpu](img.cuda())
                        cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                        # cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                        cam = cam.cpu().numpy() * labels[0].clone().view(1, 1, 1).numpy()
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam


            thread_pool = pyutils.BatchThreader(_work, list(enumerate(images)), batch_size=12, prefetch_size=0,
                                                processes=args.num_workers)
            cam_list = thread_pool.pop_results()
            sum_cam = np.sum(cam_list, axis=0)
            sum_cam[sum_cam<0] = 0
            cam_max = np.max(sum_cam, (1,2), keepdims=True)
            cam_min = np.min(sum_cam, (1,2), keepdims=True)
            sum_cam[sum_cam < cam_min+1e-5] = 0
            norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
            cam_dict = {}
            for i in range(1):
                if labels[0][i] > 1e-5:
                    cam_dict[i] = norm_cam[i]
            np.save(os.path.join(cam_dir, img_names[0] + '.npy'), cam_dict)
            print('\r# Inferring [{}/{}] = {:.2f}%'.format(iter + 1, len(infer_data_loader),
                                                           (iter + 1) / len(infer_data_loader) * 100))
