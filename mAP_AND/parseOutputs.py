import os
from os.path import join, isfile,isdir
from os import listdir
import dippykit as dip
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import re
import numpy as np

ROOT_PATH = 'C:/Users/might/Dropbox (GaTech)/Shared folders/AND_Project/testing_images/evaluation/0.35iou/'
dir_list = [f for f in listdir(ROOT_PATH) if isdir(join(ROOT_PATH,f))]
dict_list = []
""" 
This is the order they will be made in the dict_list: 
    0 = histEq_1050-histEq_input
    1 = histEq_1050-untouch_input
    2 = histEq_825-histEq_input
    3 = histEq_825-untouch_input
    4 = 'histEq_825-untouch_input-anc',
    5 = histEq_aug1050-histEq_input
    6 = histEq_aug1050-untouch_input
    7 = histEq_aug825-histEq_input
    8 = histEq_aug825-untouch_input
    9 = untouch_1050-untouch_input
    10 = untouch_825-untouch_input
    11 = untouch_aug825-histEq_input
    12 = untouch_aug825-untouch_input
"""

for count,directory in enumerate(dir_list):
    file_list = [f for f in listdir(join(ROOT_PATH,directory)) if isfile(join(ROOT_PATH,directory, f)) & f.endswith(".txt")]
    print(str(directory))
    data_dict = {}
    for output in file_list:
        # parse here
        f = open(join(ROOT_PATH,directory,output),'r')
        lines = f.readlines()
        
        # precision
        precisionline = lines[2]
        x = re.sub("Precision: \[","", precisionline)
        y = re.sub("'","", x)
        z = re.sub("]","", y)
        precision = np.fromstring(z, dtype=float, sep=',')
        # print('precision = ',precision)
        
        # recall
        recallline = lines[3]
        x = re.sub("Recall :\[","", recallline)
        y = re.sub("'","", x)
        z = re.sub("]","", y)
        recall = np.fromstring(z, dtype=float, sep=',')
        # print('recall = ',recall)

        # mAP
        mAPline = lines[1]
        x = re.sub("% = neuron AP","",mAPline)
        mAP = float(x)    
        # print('mAP = ',mAP)

        # detected objects
        detectedline = lines[13]
        x = re.sub("neuron: ","",detectedline)
        y = re.sub("\(tp:","",x)
        z = re.sub(", fp:"," ",y)
        w = re.sub("\)","",z)
        obj = np.fromstring(w,dtype=int,sep= " ")   
        tp = obj[1]
        fp = obj[2]

        data_dict = {
            "Precision" : precision,
            "Recall" : recall,
            "mAP" : mAP, 
            "TruePos" : tp,
            "FalsePos" : fp
        }

        dict_list.append(data_dict)

net_names = ['histEq_1050-histEq_input',
                'histEq_1050-untouch_input',
                'histEq_825-histEq_input',
                'histEq_825-untouch_input',
                'histEq_825-untouch_input-anc',
                'histEq_aug1050-histEq_input',
                'histEq_aug1050-untouch_input',
                'histEq_aug825-histEq_input',
                'histEq_aug825-untouch_input',
                'untouch_1050-untouch_input',
                'untouch_825-untouch_input',
                'untouch_aug825-histEq_input',
                'untouch_aug825-untouch_input']

# plt.figure(1)
# for count,net in enumerate(dict_list):
#     prec = dict_list[count]['Precision']
#     rec = dict_list[count]['Recall']
#     plt.plot(rec,prec,label=net_names[count])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Prec-Rec Plot')
# plt.show()

f1 = np.zeros(len(net_names))
for count,net in enumerate(dict_list):
    prec10 = dict_list[count]['Precision']#.astype('float')
    rec10 = dict_list[count]['Recall']#.astype('float')
    f = np.zeros(len(prec10))
    for i in range(len(prec10)):
        f[i] = 2 * (prec10[i] * rec10[i]) / (prec10[i] + rec10[i])
    f1[count] = max(f)

# print('prec',prec10[20])
# print(len(prec10))
# print(len(rec10))
plt.plot(f1)
plt.show()

# plt.figure(1)
# for count,net in enumerate(dict_list):
#     prec = dict_list[count]['Precision']
#     rec = dict_list[count]['Recall']
#     plt.plot(rec,prec,label=net_names[count])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Prec-Rec Plot')
# plt.show()
