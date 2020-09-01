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

# define camera pixel resolution
M = 3072
N = 2048
bit = 8

# Save path for all your data
save_path = 'C:/Users/Rowan_Lab/Documents/AND_Data/'

print('-----load cam-----')
# Open up camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-4)

# Testing if USB camera works
# def testDevice(source):
#     cap = cv2.VideoCapture(source)#,cv2.CAP_MSMF)
#     print(cv2.CAP_PROP_FOURCC)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
#
#     if cap is None or not cap.isOpened():
#         print('Warning: unable to open video source: ', source)
# testDevice(0)# + cv2.CAP_DSHOW) # no printout
# # testDevice(1) # prints message if no source

print('-----setup side functions-----')
# Detect the image frame using YOLO
def detect_img(yolo,img):
    # image = Image.open(img) # Only used when opening an image from a folder/ not needed for the camera
    r_image = yolo.detect_image(img)
    # r_image.show()
    return r_image
# Contrast stretching of the image for better visualization
min_range = 60 #30
max_range = 210
def contrastStretch(image, min, max):
    iI = image  # image input
    minI = min  # minimum intensity (input)
    maxI = max  # maxmimum intensity (input)
    minO = 0  # minimum intensity (output)
    maxO = 255  # maxmimum intensity (output)
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)  # image output    # for m in range(0,M):
    return iO

use_YOLO = True
my_yolo = YOLO() # start yolo session

timeLog = save_path + date + '_test_inference_time.txt'
logFileTime = open(timeLog, mode='a')
logFileTime.write("%s" % dt_string)

print('-----runnit-----')
cv2.namedWindow('live',cv2.WINDOW_NORMAL)
while(True):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, M)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, N)
    start = timer()
    # Capture frame-by-frame
    ret, frame = cap.read()

    # manipulate the frames here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # if need to grayscale
    if use_YOLO:
        alter = contrastStretch(frame, min_range, max_range)
        image = Image.fromarray(np.uint8(alter))
        # image = Image.fromarray(np.uint8(cm.gist_earth(frame)))
        output = detect_img(my_yolo, image)
        # output = predict_with_yolo_head(model, frame, config, confidence=0.3, iou_threshold=0.4)
        output = np.array(output)
        cv2.imshow('live', output)
    else:
        # alter = contrastStretch(frame, min_range, max_range)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # if need to grayscale
        # Display the resulting frame
        cv2.imshow('live', frame)
    # output_image = frame
    end = timer()
    save_time = end - start
    logFileTime.write("\n%s" % save_time)
    # print("Inference time: {:.2f}s".format(end - start))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
my_yolo.close_session() # end yolo session
logFileTime.close()