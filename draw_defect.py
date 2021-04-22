import numpy as np
import cv2
import random
import os
import json
import torch
from PIL import Image

def draw_defect(train_path, seg_path, mask_path):

    def generate_coco(coco_dict):
        image = {}
        image["height"] = coco_dict["height"]
        image["width"] = coco_dict["width"]
        image["id"] = coco_dict["img_id"]
        image["file_name"] = coco_dict["filename"]
        
        category = {}
        category["supercategory"] = "defect"
        category["id"] = 0
        category["name"] = "defect"
        
        annotation = {}
        annotation["iscrowd"] = 0
        annotation["area"] = 0.0
        annotation["image_id"] = coco_dict["img_id"]
        annotation["bbox"] = roi_range
        annotation["segmentation"] = coco_dict["segmentation"]
        annotation["category_id"] = 0 
        annotation["id"] = coco_dict["num_id"]

        return image, category, annotation

    files = os.listdir(train_path)
    
    images = []
    categories = []
    annotations= [] 
    num_id = 0
    img_id = 0
    data_coco = {}
    image = {}
    category = {}
    seg = [[]]
    # rotation and size control
    for file_ in files:   
        if file_.endswith('.jpg' or '.png'):
            im = cv2.imread(os.path.join(train_path, file_))
            mask = cv2.imread(os.path.join(mask_path, file_))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            h, w, _ = im.shape # 400, 520, 3
            im_seg = im.copy()
            contours,hierarchy = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, cnt in enumerate(contours):
                
            
                if  cv2.contourArea(cnt) > 30:
                    # Find bounding rectangles
                    x,y,w_,h_ = cv2.boundingRect(cnt)
                

                    # Draw the rectangle
                    cv2.rectangle(im,(x,y),(x+w_,y+h_),(255,255,0),1)
                    roi_range = (y,x, w_, h_)
                    cnt_ = cnt.reshape(-1)
                    
                    seg = [cnt_.tolist()]
                    coco_dict = {}
                    if len(seg[0]) > 10:
                        coco_dict = {
                        "height": h,
                        "width": w,
                        "num_id" : num_id,
                        "img_id" : img_id,
                        "roi" : roi_range,
                        "filename" : file_,
                        "segmentation" : seg
                        }

                        num_id = num_id + 1
                        image, category, annotation = generate_coco(coco_dict)
                        annotations.append(annotation)
                        cv2.drawContours(im_seg,[cnt], 0, (0,255,255), -1)
                        cv2.rectangle(im_seg,(x,y),(x+w_,y+h_),(255,255,0),1)
                cnt = []
                seg = [[]]
                roi_range = []
                cv2.imwrite(os.path.join(seg_path, file_), im_seg)    
            if image != {}:
                images.append(image)
                img_id = img_id + 1
                image = {}

        data_coco["images"] = images
        data_coco["categories"] = [category]
        data_coco["annotations"] = annotations
        #print("#####################################################",data_coco)
        json.dump(data_coco, open(train_path +'train_coco.json', "w"), indent=4)




