import numpy as np
import cv2
import math
from random import randint
def constrain(val, min_val = 0, max_val = 255):
    return min(max_val, max(min_val, val))


# im = cv2.line(img,(0,0),(27,27),(255,255,255),2)
# cv2.imshow("im",im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 1.2 1 short
length = [3,7]
width = [1,3]
color = [(0,0,255),(255,0,0)]
#ra = 0
for num in range(1000):
    print(num)
    for i in range(2): # length

        for j in range(2): # width

            for k in range(12): # angle
                angle = 15*k
                for z in range(2): # color
                    [x,y] = [randint(0+length[i],27-length[i]),randint(0+length[i],27-length[i])]
                    [x1,y1] = [x-length[i]*math.cos(angle),y-length[i]*math.sin(angle)]
                    [x2,y2] = [x+length[i]*math.cos(angle),y+length[i]*math.sin(angle)]
                    p1 = (math.ceil(x1),math.ceil(y1))
                    p2 = (math.ceil(x2),math.ceil(y2))
                    img = np.zeros((28,28,3),np.uint8)
                    im = cv2.line(img,p1,p2,color[z],width[j])
                    name = str(i)+'_'+str(j)+'_'+str(k)+'_'+str(z)+'_'+str(num+1)+".jpg"
                    cv2.imwrite(name,im)
                    #ra = ra+1

