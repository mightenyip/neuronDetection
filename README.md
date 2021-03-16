# Automated Neuron Detection using YOLOv3
By Mighten Yip and Mercedes Gonzalez ([Precision Biosystems Laboratory](http://pbl.gatech.edu/) at Georgia Tech, 2020).

This repository allows the camera (those compatible with Micro-manager) to interact with the neuron detection YOLOv3 algorithm in acute mouse brain slice. Parts of the code were cloned from /qqwweee/keras-yolo3.
Patch clamp experiments to verify the health of the neurons used Scientifica manipulator/stage and a digitally controlled [pressure control box](http://neuromaticdevices.com).

To get started: clone/fork this repository into your preferred folder location. 

live_neuron_detection.py is the origin file to run the neuron detection algorithm. Make sure the source code is updated with your specific Micro-manager config file. 

All preprocessing, training, validation, and evaluation of the neural networks was done in Python, using the following software packages:
* Python v3.6.8
* Tensorflow-gpu v1.14 (can be regular tensorflow-v1.14 if running on CPU)
* Keras v2.1.5
* Numpy v1.19.0
* Matplotlib v3.2.2
* Opencv-python v4.3.0.36
* Lxml v4.5.2
* Pillow v7.2.0
* Scikit-image v0.16.2
* Dippykit v3.0.0
* pymmcore 10.0.0.1

The network parameters used include:
1)  Optimizer: ”Adam”
2)  Learning rate: 0.001
3)  Batch size: 8
