import numpy as np
import cv2
import random
import os
import json
import torch
from PIL import image

def choose_max_part(score_map, binary_mask): # choose max pixel
    height, width, _ = binary_mask.shape
    pnt = np.argmax(score_map)
    cnts = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_mask = np.zeros((height,width,1), np.uint8)
    for c in cnts:
        result = cv2.pointPolygonTest(c, pnt, False)
        if result == 1: # inside the contours
            

def set_threshold(): # choose maximize contours

    def
    

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

def detect_how(im, mask_detect, mask_anomal_dir, file_, result_dir):
    height, width, _ = im.shape
    c,w,h= mask_detect.shape
    
    # if anomal thing not detected
    if c == 0:
        mask_detect = np.zeros((height,width,1), np.uint8)
    else:
        mask_detect  = mask_detect[0]*255
    mask_anomal = cv2.imread(os.path.join(mask_anomal_dir, file_),0)
    
    contours,hierarchy = cv2.findContours(mask_anomal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blank_image = np.zeros((height,width,1), np.uint8)
    blank_image2 = np.zeros((height,width,1), np.uint8)
    # contour 하나씩 곱해본다
    for i, cnt in enumerate(contours):
        # Find bounding rectangles
        #x, y, w_,h_ = cv2.boundingRect(cnt)
        # if the size of the contour is greater than a threshold
        #area = cv2.contourArea(cnt)
        if  cv2.contourArea(cnt) > 30:
            
            contour_image = cv2.drawContours(blank_image,[cnt], 0, 255, -1)
            # defect_image = cv2.bitwise_and(contour_image, mask_detect.astype(int))
            
            defect_image = np.multiply(contour_image.reshape((224,224)) , (255-mask_detect.reshape((224,224))))
            cv2.imwrite(result_dir + './defect/' + file_, 255-mask_detect.reshape((224,224)))
            cv2.imwrite(result_dir + './defect2/' + file_, defect_image)
            
            defect_image = (defect_image.astype(np.uint8)) *255
            cv2.imwrite(result_dir + './defect3/' + file_, defect_image)
            
            contours2,hierarchy2 = cv2.findContours(defect_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.countNonZero(contour_image)
            mask_area = cv2.countNonZero(defect_image)
            
            #if cv2.contourArea(cnt2) == area:
            for cnt2 in contours2:
                if mask_area > 0.90 * area:    
                    im = cv2.drawContours(im,[cnt2], 0, (255,255,0), -1)
                    blank_image2 = cv2.drawContours(blank_image2,[cnt2], 0, (255,255,0), -1)

        #blank_image = np.zeros((height,width,1), np.uint8)
        #blank_image2 = np.zeros((height,width,1), np.uint8)
    if not os.path.isdir(result_dir + '/result/'):
        os.mkdir(result_dir + '/result/')
    if not os.path.isdir(result_dir + '/contours/'):
        os.mkdir(result_dir + '/contours/')
    if not os.path.isdir(result_dir + '/mask/'):
        os.mkdir(result_dir + '/mask/')
    if not os.path.isdir(result_dir + '/defect/'):
        os.mkdir(result_dir + '/defect/')
    if not os.path.isdir(result_dir + '/defect2/'):
        os.mkdir(result_dir + '/defect2/')
    if not os.path.isdir(result_dir + '/defect3/'):
        os.mkdir(result_dir + '/defect3/')
    
    cv2.imwrite(result_dir + '/result/' + file_, im)
    cv2.imwrite(result_dir + '/contours/' + file_, blank_image2)
    cv2.imwrite(result_dir + '/mask/' + file_, mask_detect)
