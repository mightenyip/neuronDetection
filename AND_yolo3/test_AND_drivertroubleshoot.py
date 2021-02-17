"""
    ECE 6258: Digital Image Processing
    AND Project Code

    Mighten Yip
    Mercedes Gonzalez

    Purpose: Script to test neural network

    make sure that paths in yolo.py to model, classes, anchors, 
    and font are all absolute paths on your machine

    reads images in from root_path and saves annotated pngs to save_path
"""
import sys
from yolo import YOLO, detect_video
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os 

def detect_img(yolo,img):
    image = Image.open(img)
    r_image = yolo.detect_image(image)
    r_image.show()
    return r_image

img = 'C:/Users/myip7/Documents/GitHub/neuronDetection_YOLO/slice4j.png'
# Set paths here
# root_path = 'C:/Users/myip7/Dropbox (GaTech)/Shared folders/AND_Project/FINAL_CODE/Images/'
# save_path = 'C:/Users/myip7/Documents/AND_Data/'
# save_time = 'C:/Users/myip7/Dropbox (GaTech)/Shared folders/AND_Project/FINAL_CODE/Evaluation/inference/'

# file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(".png")]
my_yolo = YOLO() # start yolo session

# for f in file_list:
    # base = os.path.basename(f)
    # logID = save_path + os.path.splitext(base)[0] + '.txt'
    # net = os.path.dirname(save_path)
    # logTime = save_time + os.path.basename(net) + '_inference_time.txt'
annotated = detect_img(my_yolo,img)
    # annotated.save(join(save_path,f),"png")

my_yolo.close_session() # end yolo session
