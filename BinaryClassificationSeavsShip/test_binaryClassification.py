from unet_binaryClassification import *
from data_binaryClassification import *
import os
import numpy as np
import scipy.misc

mydata = dataProcess(128,128)

print("loading data")
imgs_test, labels_test = mydata.load_test_data(2)
print("loading data done")

mycnn = myCNN(128,128)
model = mycnn.get_CNN()
print("got CNN")

model.load_weights('CNNweights.hdf5')
print("loaded model weights")

imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
loss, acc = model.evaluate(imgs_test, labels_test, batch_size=1, verbose=1)
print('loss on test data is ', loss)
print('acc on test data is ', acc)
print("saving test masks")
np.save('npydata/imgs_mask_test.npy', imgs_mask_test)

print("array to image")
imgs_true = labels_test
print('loading done, now saving')
for i in range(imgs_true.shape[0]):
    img = imgs_test[i,:,:,0]
    img_pred = imgs_mask_test[i,:,:,0]
    img_true = imgs_true[i,:,:,0]
    if not os.path.lexists('results'):
        os.mkdir('results')
    scipy.misc.imsave("results/{}_img.jpg".format(i), img)
    scipy.misc.imsave("results/{}_pred.jpg".format(i), img_pred)
    scipy.misc.imsave("results/{}_true.jpg".format(i), img_true)