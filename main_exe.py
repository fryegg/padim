import subprocess
import numpy as np
from multiprocessing import Process
import os

# Basic Python Environment
python = "C:\\Users\\sss_sim2\\miniconda3\\envs\\myenv\\python"

"""
parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
parser.add_argument('--save_path', type=str, default='./mvtec_result')
parser.add_argument('--test', type=str, default='train', choices = 'train, test')
parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
"""
data_dir = './'
size = 256
load = './mvtec_result'
arch = 'wide_resnet50_2'
gt_result = './gt_result'
nogt_result = './nogt_result'
# tensorboard --logdir=runs

def main_script():
    gt = 'gt'
    script1 = '%s main.py --data_path %s --gt %s --save_path %s' % (python, data_dir, gt, gt_result)
    #subprocess.call(script1, shell=True)
    script2 = '%s hello.py --gt %s' % (python, gt)
    subprocess.call(script2, shell=True)
    
    gt = 'nogt'
    #script1 = '%s main.py --data_path %s --gt %s --save_path %s' % (python, data_dir, gt, nogt_result)
    #subprocess.call(script1, shell=True)
    script2 = '%s hello.py --gt %s' % (python, gt)
    subprocess.call(script2, shell=True)

main_script()