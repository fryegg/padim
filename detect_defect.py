import numpy as np
import cv2
import random
import os
import json

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
