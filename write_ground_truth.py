import numpy as np
import cv2
import random
import os
import json
import torch
from PIL import Image
import itertools


ground_truth = '2/ground_truth/'
json_path = '2/test/json/'
img = cv2.imread('2/train/good/2021-04-14 140805.jpg')
#img = cv2.imread('1/ground_truth/defect/2021-01-28 145336.jpg',0)
height, width, _ = img.shape
files = os.listdir(json_path)
defect_image = []

"""
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    print(cnt[0])
"""
for json_file in files:
    file_name = os.path.splitext(json_file)[0]
    blank_image =  np.zeros((height,width), np.uint8)
    with open(os.path.join(json_path, json_file)) as f:
        data = json.load(f)
        print(len(data["shapes"]))
        
        for shapes in (data["shapes"]):
            cnts = shapes["points"]
            for i,cnt in enumerate(cnts):
                cnts[i] = cnt
            print(np.array(cnts).astype(int))
            
            cv2.drawContours(blank_image,[np.array(cnts).astype(int)], 0, 255, -1)
        cv2.imwrite(ground_truth+file_name+'.jpg', blank_image)
