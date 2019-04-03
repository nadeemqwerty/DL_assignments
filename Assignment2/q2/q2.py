from __future__ import print_function
import numpy as np

batch_size = 32
epochs = 12
import keras
from keras.models import Model,Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GaussianNoise, Input, Dropout, concatenate
from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import keras.backend as K
from keras.backend import tf as ktf


data_dir = "../input/"
(x_train, y_train_length, y_train_width, y_train_color, y_train_angle) = (np.load(data_dir +'x_train.npy'), np.load(data_dir+'y_train_length.npy'), np.load(data_dir+'y_train_width.npy'), np.load(data_dir+'y_train_color.npy'), np.load(data_dir+'y_train_angle.npy'))
(x_test, y_test_length, y_test_width, y_test_color, y_test_angle) = (np.load(data_dir+'x_test.npy'), np.load(data_dir+'y_test_length.npy'), np.load(data_dir+'y_test_width.npy'), np.load(data_dir+'y_test_color.npy'), np.load(data_dir+'y_test_angle.npy'))

def inception_block(x, filters):
#     last = x

    net1 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)

    net2 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)
    net2 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net2)

    net3 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)
    net3 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net3)
    net3 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net3)

    output = concatenate([net1, net2, net3], axis=3)
    return output

input_tensor = Input(shape=(28, 28, 3))

x = Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation = 'relu')(input_tensor)
x = Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(5, 5), strides=(2,2))(x)

x = inception_block(x,32)
x = MaxPool2D(pool_size=(3, 3), strides=(1,1),padding='Same')(x)
x = Flatten()(x)

length = Dense(128, activation = 'sigmoid')(x)
length = Dropout(0.25)(length)
length = Dense(1, activation = 'sigmoid',name = 'length')(length)

width = Dense(1, activation = 'sigmoid',name = 'width')(x)

color = Dense(1, activation = 'sigmoid',name = 'color')(x)

angle = Dense(256, activation = 'softmax',name = 'angle')(x)
angle = Dropout(0.5)(x)
angle = Dense(12, activation = 'softmax',name = 'angle')(angle)

model = Model(input_tensor,[length, width, color, angle] )
model.summary()

loss = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy']
loss_weights = [0.1, 0.1, 0.1, 2.0]
model.compile(optimizer='adam',
              loss=loss,
              loss_weights = loss_weights,
              metrics=['accuracy'])

history = model.fit(x_train, [y_train_length, y_train_width, y_train_color, y_train_angle], epochs=epochs, batch_size=batch_size, shuffle=True,verbose=1)
model.save_weights("line_2_final.h5")

model.evaluate(x_test, [y_test_length, y_test_width, y_test_color, y_test_angle])



def plot_history(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(hist['epoch'], hist['length_loss'],
	       label='Length Loss')
	plt.plot(hist['epoch'], hist['width_loss'],
	       label='Width Loss')
	plt.plot(hist['epoch'], hist['color_loss'],
	       label='Color Loss')
	plt.plot(hist['epoch'], hist['angle_loss'],
	       label='Angle Loss')
	plt.plot(hist['epoch'], hist['loss'],
	       label='Train Loss')
# 	plt.ylim([0,1])
	plt.legend()
	plt.savefig('loss_plot.png')

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')

	plt.plot(hist['epoch'], hist['length_acc'],
	       label='Length Acc')
	plt.plot(hist['epoch'], hist['width_acc'],
	       label='Width Acc')
	plt.plot(hist['epoch'], hist['color_acc'],
	       label='Color Acc')
	plt.plot(hist['epoch'], hist['angle_acc'],
	       label='Angle Acc')
# 	plt.ylim([0,1])
	plt.legend()
	plt.savefig('acc_plot.png')
#


plot_history(history)

keras.utils.plot_model(
    model,
    to_file='model.png',
    show_shapes=True
)

pred = model.predict(x_test)

l = pred[0]
l[l>=0.5] = 1
l[l<0.5] = 0


cm_l = confusion_matrix(y_test_length,l)
cm_l
np.save("confusion_matrix_l.npy",cm_l)
plt.matshow(cm_l)
plt.colorbar()
plt.savefig("confusion_l.jpg")



w=pred[1]
w[w>=0.5] = 1
w[w<0.5] = 0

cm_w = confusion_matrix(y_test_width,w)
np.save("confusion_matrix_w.npy",cm_w)
plt.matshow(cm_w)
plt.colorbar()
plt.savefig("confusion_w.jpg")



c = pred[2]
c[c>=0.5] = 1
c[c<0.5] = 0
cm_c = confusion_matrix(y_test_color,c)
np.save("confusion_matrix_c.npy",cm_c)
plt.matshow(cm_c)
plt.colorbar()
plt.savefig("confusion_c.jpg")



a = []
for i in pred[3]:
    x=np.argmax(i)
    a.append(x)
a = np.array(a)

cm_a = confusion_matrix(y_test_angle,a)
np.save("confusion_matrix_a.npy",cm_a)
plt.matshow(cm_a)
plt.colorbar()
plt.savefig("confusion_a.jpg")
