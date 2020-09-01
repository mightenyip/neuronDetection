from os import listdir
from os.path import isfile, join
import os
import sys
import argparse
from yolo import YOLO#, detect_video
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pymmcore
from timeit import default_timer as timer
from datetime import datetime

now = datetime.now() # datetime object containing current date and time
print("now =", now)
dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
date = now.strftime("%b-%d-%Y")
M = 1280
N = 1024

save_path = 'C:/Users/myip7/Documents/AND_Data/'

def detect_img(yolo,img):
    # image = Image.open(img) # Only used when opening an image from a folder/ not needed for the camera
    r_image = yolo.detect_image(img)
    # r_image.show()
    return r_image

def contrastStretch(image, min, max):
    iI = image  # image input
    minI = min  # minimum intensity (input)
    maxI = max  # maxmimum intensity (input)
    minO = 0  # minimum intensity (output)
    maxO = 255  # maxmimum intensity (output)
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)  # image output    # for m in range(0,M):
    # M = 1280
    # N = 1024
    # for m in range(0,M):
    #     for n in range(0,N):
    #         if minI < iI[m,n] < maxI:
    #             iI[m,n] = iI[m,n]
    #         elif iI[m,n] < minI:
    #             iI[m,n] = 0
    #         elif iI[m,n] > maxI:
    #             iI[m,n] = 255
    # if iI < minI:
    #     iO = 0
    # if iI > maxI:
    #     iO = 255
    return iO

mmc = pymmcore.CMMCore()
print('-----setup cam-----')
mm_dir = 'C:/Program Files/Micro-Manager-2.0gamma/'
mmc.setDeviceAdapterSearchPaths([mm_dir])
print(mmc.getVersionInfo())
print(mmc.getAPIVersionInfo())

use_YOLO = True
my_yolo = YOLO() # start yolo session

print('-----load cam-----')
# print(os.path.join(mm_dir, 'MMConfig_1.cfg'))
mmc.loadSystemConfiguration(mm_dir + 'MMConfig_QCam.cfg')
mmc.setExposure(200)
mmc.snapImage()
im1 = mmc.getImage()

# print('----snap an image-----')
min_range = 55 #30
max_range = 80
plt.figure(1)
conStretch_vec = np.vectorize(contrastStretch)
img = conStretch_vec(im1, min_range, max_range)
plt.subplot(2,1,1)
plt.imshow(im1,'gray')
plt.subplot(2,1,2)
plt.hist(img,10)
plt.show()

# print('-----runnit-----')
timeLog = save_path + date + '_test_inference_time.txt'
logFileTime = open(timeLog, mode='a')
logFileTime.write("%s" % dt_string)

cv2.namedWindow('live',cv2.WINDOW_AUTOSIZE)
mmc.startContinuousSequenceAcquisition(1)
while True:
    if mmc.getRemainingImageCount() > 0:
        start = timer()
        # frame = mmc.getLastImage()
        # print(frame)
        # print(frame.shape)
        frame = mmc.popNextImage()
        # Specify the min and max range
        # frame = conStretch_vec(frame, min_range, max_range)
        # Run detection
        if use_YOLO:
            alter = contrastStretch(frame, min_range, max_range)
            image = Image.fromarray(np.uint8(alter))
            # image = Image.fromarray(np.uint8(cm.gist_earth(frame)))
            output = detect_img(my_yolo,image)
            # output = predict_with_yolo_head(model, frame, config, confidence=0.3, iou_threshold=0.4)
            output = np.array(output)
            cv2.imshow('live',output)
        else:
            cv2.imshow('live', frame)
        # output_image = frame
        end = timer()
        save_time = end-start
        logFileTime.write("\n%s" % save_time)
        # print("Inference time: {:.2f}s".format(end - start))

    if cv2.waitKey(1) & 0xFF == ord('q'): # This break key is critical, otherwise the live image does not load
        break
# Close opencv window, MMC session, YOLOv3 session, and inference time log
cv2.destroyAllWindows()
mmc.stopSequenceAcquisition()
mmc.reset()
my_yolo.close_session() # end yolo session
logFileTime.close()

# print('-----hope you enjoyed the show-----')