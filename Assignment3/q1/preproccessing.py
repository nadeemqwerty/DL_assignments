import os
import numpy as np
import cv2
import pandas as pd
target_shape = (128,128,3)

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


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

classes = ['Knuckle', 'Palm', 'Vein']
data_dir = '../data/Q1/task1/'

images = []
boxes = []
labels = []

for i in range(3):
    label = np.zeros(3)
    label[i] = 1

    dataframe = pd.read_csv(data_dir+classes[i]+"/groundtruth.txt")

    _labels = [label]*dataframe.shape[0]
    labels = _labels + labels

    paths = list(dataframe['Path'])

    for j,file in enumerate(paths):
        if i == 0:
            a = file.split()
            file = a[0]+'_'+a[1]
            print(file)
        print(data_dir+classes[i]+'/'+file)
        img = cv2.imread(data_dir+classes[i]+'/'+file)
        box = list(dataframe.loc[j][1:5])
        img, box = resize(img,box,target_shape)
        images.append(img)
        boxes.append(box)
images = np.array(images)
boxes = np.array(boxes)
labels = np.array(labels)

print(images.shape)
print(labels.shape)
print(boxes.shape)
images, boxes, labels = unison_shuffled_copies(images,boxes,labels)
np.save("images.npy",images)
np.save("boxes.npy",boxes)
np.save("labels.npy",labels)
