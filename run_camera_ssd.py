### Need to transfer trained model from Jetson Nano. CLASS NUMBER 2
# SSD Detection code working fine on PC
### https://github.com/qfgaohao/pytorch-ssd



import gi
gi.require_version('Gtk', '2.0')



from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

import cv2
import sys
import matplotlib.pyplot as plt


# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image


import time
from timeit import default_timer as timer




net = create_mobilenetv1_ssd(2, is_test=True) # 2 classes

net.load('models/soy-best-ssd-ep35-loss3_71.pth')

predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""
def draw_bboxes(image, bboxes): ###results
###bboxes = results[image_idx]
    for idx in range(len(bboxes)):
        # get the bounding box coordinates in xyxy format
        x1, y1, x2, y2 = bboxes[idx]
        # resize the bounding boxes from the normalized to 300 pixels
        x1=abs(int(x1))
        y1=abs(int(y1))
        x2=abs(int(x2))
        y2=abs(int(y2))

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA
        )

    plt.imshow(image)
    plt.axis('off')
###    plt.show() ### Just to show the RT pic

    return image


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=256, #960,
    display_height=256, #540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

start = timer()
def show_camera():
    window_title = "SimCam"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        ###try:
            ###window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        while True:
                
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                j = timer() - start #range(5)
                i = int(j*300)
        ###        if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                ret_val, frame = video_capture.read()
                    ###cv2.imshow(window_title, frame) ### Live streamimg !
                frm = frame
                image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                boxes, labels, probs = predictor.predict(image, 10, 0.4)
                bi=boxes.cpu().detach().numpy().astype(int) ### tensor to float, then float to int ### ONLY NEEDED FOR BOUNDING BOX
                print(bi)
                draw_bboxes(frm, bi) 
                cv2.imwrite("s_cam/simple"+str(i)+".jpg", frm)
                      
           ###     else:
              ###      break 
                ### keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                ###if keyCode == 27 or keyCode == ord('q'):
                   ### break
        ###finally:
           ### video_capture.release()
            ###cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
###plt.box(False)




# class_names = [name.strip() for name in open(label_path).readlines()]



### im = cv2.imread('soy_data/test/341.JPG')
# draw_bboxes(im, boxx) ### for e32, boxx is not working, but bi is !!!

