from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from libs.pascal_voc_io import PascalVocReader,PascalVocWriter
from libs.pascal_voc_io import XML_EXT
import math
from PIL import Image
from os.path import join, isfile
from os import listdir
from xml.etree import ElementTree as et

# FUNCTION DEFINITIONS _____________________________________________
def changeXML(xmlPath,name,root_path):
    if xmlPath is None:
        return
    if os.path.isfile(xmlPath) is False:
        return
    tree = et.parse(xmlPath)
    tree.find('.//filename').text = name + '.png'
    tree.find('.//path').text = root_path + name + '.png'
    tree.write(xmlPath)

# Set path where all the images are, get list of all tiff files in that dir
root_path = "C:/Users/mgonzalez91/Dropbox (GaTech)/Coursework/SU20 - Digital Image Processing/AND_Project/orig/full_untouched/"
file_type = ".xml"
file_list = [f for f in listdir(root_path) if isfile(join(root_path, f)) & f.endswith(file_type)]

# CROPPING
for count, filename in enumerate(file_list):
    base,ext = os.path.splitext(filename)
    fullfile = join(root_path,filename)
    changeXML(fullfile,base,root_path)

    print(count)
