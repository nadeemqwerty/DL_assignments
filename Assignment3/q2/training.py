#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from network import net
# from keras.preprocessing.image import ImageDataGenerator


# In[2]:


data_dir = "data/"


# In[3]:


images = np.load(data_dir+"new_images.npy")
masks = np.load(data_dir+"new_masks.npy")//255.


# In[4]:


images.shape


# In[5]:


masks.shape


# In[6]:


masks = masks.reshape((10000,320,416,1))


# In[7]:


masks.shape


# In[8]:


# datagen = ImageDataGenerator(
#     rotation_range=25,
#     rescale = 1./255.,
#     vertical_flip = True ,
#     horizontal_flip = True,
#     width_shift_range=0.1,
#     shear_range = 0.1,
#     zoom_range = 0.1,
#     height_shift_range=0.1)


# In[9]:


model = net((320,416,3))
model.summary()


# In[10]:


# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# In[11]:


# for batch in datagen.flow(x, batch_size=1, seed=1337): img.append(batch)
# for batch in datagen.flow(ymask, batch_size=1, seed=1337): mask.append(batch)


# In[14]:


epochs = 2
batch_size = 32


# In[15]:


# histories = []
# for epoch in range(epochs):
#     print(epoch)
#     img = []
#     mask = []
    
# #     for i in range(batch_size):
#     for batch in datagen.flow(x, batch_size=1, seed=1337): img.append(batch)
#     for batch in datagen.flow(ymask, batch_size=1, seed=1337): mask.append(batch)
#     history = model.fit(img, mask, epochs=1, batch_size=batch_size, shuffle=True,verbose=1)
#     histories.a
history = model.fit(images, masks, epochs=epochs, batch_size=batch_size, shuffle=True,verbose=1)


# In[ ]:




