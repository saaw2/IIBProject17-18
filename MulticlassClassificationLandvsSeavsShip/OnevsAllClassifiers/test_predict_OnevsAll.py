from unet_OnevsAll import *
from data_OnevsAll import *
import numpy as np
import os
import scipy.misc

mydata = dataProcess(128,128)

mycnn = myCNN(128,128)
model = mycnn.get_CNN()
print("got CNN")
masks_true = []
masks_pred = []

for mask in range(3):
    model.load_weights(str(mask) + '/CNNweights.hdf5')
    print("loaded model weights for mask " + str(mask))
    print("loading data")
    imgs_test, labels_test = mydata.load_test_data(mask)
    masks_true.append(labels_test)
    print("loading data done")
    masks_pred.append(model.predict(imgs_test, batch_size=1, verbose=1))
    loss, acc = model.evaluate(imgs_test, labels_test, batch_size=1, verbose=1)
    print('loss on test data for mask', str(mask), 'is ', loss)
    print('acc on test data for mask', str(mask), 'is ', acc)

imgs_true = np.concatenate(masks_true, axis=3)
imgs_mask_test = np.concatenate(masks_pred, axis=3)
print("saving test masks")
np.save('npydata/imgs_mask_test.npy', imgs_mask_test)

print("array to image")
print('loading done, now saving')
for i in range(imgs_true.shape[0]):
    img = imgs_test[i,:,:,0]
    img_pred = imgs_mask_test[i,:,:,:]*255
    img_true = imgs_true[i,:,:,:]*255
    if not os.path.lexists('results'):
        os.mkdir('results')
    scipy.misc.imsave("results/{}_img.png".format(i), img)
    scipy.misc.imsave("results/{}_pred.png".format(i), img_pred)
    scipy.misc.imsave("results/{}_true.png".format(i), img_true)
    # save test results for each One-vs_All classifier in their respective folders
    for mask in range(3):
        _img_pred = imgs_mask_test[i, :, :, mask] * 255
        _img_true = imgs_true[i, :, :, mask] * 255
        savedir = str(mask) + "/results"
        if not os.path.lexists(savedir):
            os.mkdir(savedir)
        scipy.misc.imsave(savedir + "/{}_img.png".format(i), img)
        scipy.misc.imsave(savedir + "/{}_pred.png".format(i), _img_pred)
        scipy.misc.imsave(savedir + "/{}_true.png".format(i), _img_true)