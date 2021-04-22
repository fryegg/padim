import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec
import math
import itertools
from draw_defect import draw_defect
from detector import training
from sklearn.metrics import f1_score
import csv
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def write_csv(file, text):

    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['name','precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'name':text[0],'precision': text[1], 'recall': text[2]})

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--class_name', type=str, choices=['1', '4'], default='1')
    parser.add_argument('--gt', type=str, choices=['gt', 'nogt'], default='gt')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    class_name = mvtec.CLASS_NAMES[0]
    if args.gt == 'nogt':
        #training(class_name + '/origin_train/', class_name + '/origin_test/', class_name + '/exp/', '', class_name + '/' + 'gt' + 'pred/', class_name)
        training(class_name + '/origin_train/', class_name + '/origin_test/', class_name + '/exp/', 'a', class_name + '/' + 'gt' + 'pred/', class_name)
    filelist = os.listdir(class_name + '/exp/contours/')
    filelist = np.sort(filelist)
    result_imgs = []
    TPRs = []
    TNRs = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    f1s = []
    gt_test_imgs = []
    jac_score_mean = np.zeros((224,224))

    for num, (file_) in enumerate(filelist):
        if args.gt == 'nogt':
            result_img = cv2.imread(os.path.join(class_name + '/exp/contours/', file_), 0)
        else:
            result_img = cv2.imread(os.path.join(class_name + '/gtpred/', file_), 0)
        
        gt_test_img = cv2.imread(os.path.join(class_name + '/segment/', file_), 0)
        """
        gt_test_img = cv2.imread(os.path.join(class_name + '/ground_truth/defect', file2_), 0)
        gt_test_img = cv2.resize(gt_test_img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        gt_test_img = center_crop(gt_test_img, (224,224))
        """    
        gt_test_img = np.array(gt_test_img)
        result_img = np.array(result_img)
    
        gt_test_img[gt_test_img > 0] = 1
        gt_test_img[gt_test_img <= 0] = 0
        
        result_img[result_img > 0] = 1
        result_img[result_img <= 0] = 0

        gt_test_imgs.append(gt_test_img)
        result_imgs.append(result_img)
        print(gt_test_img)
        precision = precision_score(gt_test_img.flatten(), result_img.flatten(), average='binary', zero_division=1)
        recall = recall_score(gt_test_img.flatten(), result_img.flatten(), average='binary', zero_division=1)
        # # true negative rate
        # TNR = jaccard_score(1-gt_test_img.flatten(), 1-result_img.flatten())
        # # true positive rate(recall)
        # TPR = jaccard_score(gt_test_img.flatten(), result_img.flatten())
        # TPRs.append(TPR)
        # TNRs.append(TNR)
        # f_score = f1_score(gt_test_img.flatten(), result_img.flatten())
        # f1s.append(f_score)
        write_csv(class_name + args.gt + '.csv', [file_, precision, recall])

    gt_test_imgs = np.array(gt_test_imgs)
    result_imgs = np.array(result_imgs)
    
    # # true negative rate
    # TNR = jaccard_score(1-gt_test_imgs.flatten(), 1-result_imgs.flatten())
    # # true positive rate(recall)
    # TPR = jaccard_score(gt_test_imgs.flatten(), result_imgs.flatten())
    # f_score = f1_score(gt_test_imgs.flatten(), result_imgs.flatten())
    precision = precision_score(gt_test_imgs.flatten(), result_imgs.flatten(), average='binary', zero_division=1)
    recall = recall_score(gt_test_imgs.flatten(), result_imgs.flatten(), average='binary', zero_division=1)
    write_csv(class_name + 'avg' + args.gt + '.csv', [file_, precision, recall])
    
    #jac_score = jaccard_similarity_score(gt_test_imgs, result_imgs, Normalize = True/False)
    #jac_score_mean = jac_score_mean/num