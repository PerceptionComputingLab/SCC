import argparse
import torch
from networks.vnet_center import VNet
import h5py,cv2
import math
import nibabel as nib
import numpy as np
from medpy import metric
from utils.measures import dc,jc, hd95,asd,hd
from surface_distance import metrics as surf_metric
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict
from skimage.measure import label

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

"""
Testing
Ref:
Shape-Aware Organ Segmentation by Predicting Signed Distance Maps
https://arxiv.org/abs/1912.03849
"""

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='data path')
parser.add_argument('--model', type=str,  default='2022-05-28-15-22-04', help='model_name')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
parser.add_argument('--normalization', type=str,  default='batchnorm', help='normalization to use')
parser.add_argument('--epoch_num', type=int,  default='6000', help='checkpoint to use')
parser.add_argument('--spacing', type=float,  default=1, help='voxel space')
parser.add_argument('--post', type=int, default=True, help='whether use cca')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


snapshot_path = "../semi_model/SCC/"+FLAGS.model+"/"
test_save_path = "../semi_model/prediction/"+FLAGS.model+"/"

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

patch_shape=(112, 112,80)

with open( '../data/test.list', 'r') as f:
    image_list = f.readlines()# [:4]
    image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes, normalization=FLAGS.normalization,has_dropout=False,has_triup=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()


    avg_metric = dist_test_all_case(net, image_list, num_classes=num_classes,
                                        save_result=False, test_save_path=test_save_path,
                                        has_post=FLAGS.post)


    return avg_metric


def dist_test_all_case(net, image_list, num_classes, save_result=True, test_save_path=None, has_post=False):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['asd'] = list()
    metric_dict['95hd'] = list()
    metric_dict['hd'] = list()
    metric_dict['nsd'] = list()
    
    for image_path in tqdm(image_list):
        case_name = image_path.split('/')[-2]
        id = image_path.split('/')[-1]
        
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        with torch.no_grad():
            prediction, score_map = test_single_case_patch(net, image, 18,18, 4, patch_shape , num_classes=num_classes)
            
        if np.sum(prediction) == 0:
            single_metric = (0,0,0,0,0,0)
        else:
            if has_post:
                print('post')
                prediction = getLargestCC(prediction)
            single_metric = calculate_metric_percase(prediction, label[:], space=FLAGS.spacing)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(single_metric[0])
            metric_dict['jaccard'].append(single_metric[1])
            metric_dict['asd'].append(single_metric[2])
            metric_dict['95hd'].append(single_metric[3])
            metric_dict['hd'].append(single_metric[4])
            metric_dict['nsd'].append(single_metric[5])
            print(case_name,single_metric)


        # total_metric += np.asarray(single_metric)

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), np.eye(4)), test_save_path_temp + '/' + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), test_save_path_temp + '/' +  id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label.astype(np.uint8), np.eye(4)), test_save_path_temp + '/' + id + "_gt.nii.gz")
    
    avg_metric = total_metric / len(image_list)
    metric_csv = pd.DataFrame(metric_dict)
    if has_post:
        metric_csv.to_csv(test_save_path + '/metric_post_' + str(FLAGS.epoch_num) + '.csv', index=False)
    else:
        metric_csv.to_csv(test_save_path + '/metric_'+str(FLAGS.epoch_num)+'.csv', index=False)
    print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_single_case_patch(net, image, stride_x,stride_y, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    

    for x in range(0, sx):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                y_1, y_2 = net(test_patch)
                
                y_1_soft = F.softmax(y_1, dim=1)
                y_2_soft = F.softmax(y_2, dim=1)
                y = torch.mean(torch.stack([y_1_soft, y_2_soft]), dim=0)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
                
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt, space=0.625):
    dice = dc(pred, gt)
    jc_score = jc(pred, gt)
    dice_medpy = metric.binary.dc(pred,gt)
    jc_medpy = metric.binary.jc(pred,gt)
    print(dice_medpy,jc_medpy)
    asd_score = asd(pred, gt, voxelspacing=space)
    hd95_score = hd95(pred, gt, voxelspacing=space)
    hd_score = hd(pred, gt, voxelspacing=space)
    nsd = normalized_surface_dice(pred, gt, voxelspacing=space)
    return dice, jc_score, asd_score, hd95_score, hd_score, nsd


def normalized_surface_dice(pred, gt, voxelspacing=None):
    surface_dis = surf_metric.compute_surface_distances(gt.astype(np.bool),
                                                        pred.astype(np.bool),
                                                        spacing_mm=(voxelspacing, voxelspacing, voxelspacing))
    surface_dice = surf_metric.compute_surface_dice_at_tolerance(surface_dis, 1)
    return surface_dice


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC=np.array(largestCC,dtype=np.int)
    
    return largestCC


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.epoch_num)
    