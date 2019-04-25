import os
import numpy as np
import cv2
# target_shape = (128,128,3)

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

img_files = os.listdir("../data/Q2/Data")
mask_files = os.listdir("..data/Q2/Mask")

img_prefix = img_files[0][:15]
mask_prefix = mask_files[0][:23]


images = []
masks = []
for file in img_files:
    img = cv2.imread("../data/Q2/Data/"+file)
    mask_file = mask_prefix + file[15:]
    mask = cv2.imread("../data/Q2/Mask/"+mask_file,cv2.IMREAD_UNCHANGED)
    images.append(img)
    masks.append(mask)
images = np.array(images)
# images.shape

masks = np.array(masks)
# masks.shape

np.save("data/masks.npy",masks)
np.save("data/images.npy",images)

# masks = np.load("masks.npy")
# images = np.load("images.npy")

new_masks = []
new_images = []
for i in range(10000):
    image = cv2.copyMakeBorder(images[i], 10, 10, 8, 8, cv2.BORDER_CONSTANT, value=[0,0,0])
    mask = cv2.copyMakeBorder(masks[i], 10, 10, 8, 8, cv2.BORDER_CONSTANT, value=[0])
    new_masks.append(mask)
    new_images.append(image)

new_masks = np.array(new_masks)
# new_masks.shape
np.save("data/new_masks.npy",new_masks)

images = np.array(new_images)
# images.shape
np.save("data/new_images.npy",images)
