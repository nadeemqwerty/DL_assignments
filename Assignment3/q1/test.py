# import os
import numpy as np
import cv2
import pandas as pd
import time

images = np.load("images.npy")
boxes = np.load("boxes.npy")
labels = np.load("labels.npy")


boxes = np.asarray(boxes, dtype=np.int)
# np.save("boxes.npy",boxes)
# np.save("labels.npy",labels)
target_shape = (128,128,3)

def resize(img,box, target_shape, aspectRatio = True):
    if aspectRatio:
        old_size = img.shape[:2]

        ratio = min(float(target_shape[0])/float(img.shape[0]), float(target_shape[1])/float(img.shape[1]))

        d1 = int(img.shape[1]*ratio)
        d0 = int(img.shape[0]*ratio)

        img = cv2.resize(img, (d1,d0) )

        delta_w = target_shape[1] - d1
        delta_h = target_shape[0] - d0

        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        new_box = ratio*np.asarray(box,dtype = np.float32)

        new_box[0],new_box[1] = new_box[0]+delta_w/2, new_box[1]+delta_h/2
        new_box[2],new_box[3] = new_box[2]+delta_w/2, new_box[3]+delta_h/2

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im, new_box

    return cv2.resize(img, (target_shape[1],target_shape[0]))

# data_dir = '../data/Q1/task1/'

# dataframe = pd.read_csv(data_dir+"Knuckle/groundtruth.txt")

# file = dataframe['Path'][0]

# a = file.split()
# file = a[0]+'_'+a[1]
# print(file)
# imreal = cv2.imread(data_dir+"Knuckle/"+file)
# box_real = list(dataframe.loc[0][1:5])
a = 1
# img,boxes = resize(imreal,box_real,target_shape)
for a in range(100):
    img = cv2.rectangle(images[a], (boxes[a][0], boxes[a][1]), (boxes[a][2], boxes[a][3]), (255,0,0), 2)
    cv2.imshow("lalala", img)
    # time.sleep(0.2)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    # cv2.imshow("real", imreal)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    #     break
# img = cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255,0,0), 2)
# imreal = cv2.rectangle(imreal, (box_real[0], box_real[1]), (box_real[2], box_real[3]), (255,0,0), 2)

 # 0==wait forever
cv2.destroyAllWindows()
