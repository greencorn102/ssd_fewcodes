### Need to transfer trained model from Jetson Nano. CLASS NUMBER 2
# SSD Detection code working fine on PC
### https://github.com/qfgaohao/pytorch-ssd

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

import cv2
import sys
import matplotlib.pyplot as plt
###plt.box(False)

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
    plt.show()
    return image


# class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv1_ssd(2, is_test=True) # 2 classes

net.load('new244.pth')


predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

im = cv2.imread('data_new/test/simple54597.jpg')
image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

bi=boxes.cpu().detach().numpy().astype(int) ### tensor to float, then float to int ### ONLY NEEDED FOR BOUNDING BOX
print(bi)
### Non max supression ###

#boxx=bi[probs>.95]

### Non max

draw_bboxes(im, bi) # draw_bboxes(im, boxx) ### for e32, boxx is nor working, but bi is !!!

### GT box
"""
gt=[] # from excel csv file [] if no value, otherwise [[]]
def db_bl(image, bboxes): ###results
###bboxes = results[image_idx]
    for idx in range(len(bboxes)):
        # get the bounding box coordinates in xyxy format
        x1, y1, x2, y2 = bboxes[idx]
        # resize the bounding boxes from the normalized to 300 pixels
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
        )

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image

db_bl(im, gt)
"""

### plt.savefig('bb2.png', bbox_inches='tight',pad_inches = 0)

#print(boxx)
