import cv2
import numpy as np
import os

target_shape = (128,128,1)
data_dir = "../data/Q3/Core_Point/"
def resize(img, point, target_shape, aspectRatio = True):
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

        new_point = ratio*np.asarray(point,dtype = np.float32)

        new_point[0],new_point[1] = new_point[0]+delta_w/2, new_point[1]+delta_h/2
#         new_box[2],new_box[3] = new_box[2]+delta_w/2, new_box[3]+delta_h/2

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im, new_point
    return cv2.resize(img, (target_shape[1],target_shape[0]))

img_files = os.listdir(data_dir+"Data")
point_files = os.listdir(data_dir+"Ground_truth")

point_suffix = point_files[0][-7:]

images = []
points = []
for file in img_files:
    img = cv2.imread(data_dir+"Data/"+file)
    point_file = file[:-5]+point_suffix
    file = open(data_dir+"Ground_truth/"+point_file)
    f = file.read()
    p = f.split()
    point = [int(p[0]),int(p[1])]
    img, point = resize(img,point,target_shape)
    images.append(img)
    points.append(point)
images = np.array(images)
points = np.array(points)

print(images.shape)
print(points.shape)

np.save("images_128.npy", images)
np.save("points_128.npy", points)
