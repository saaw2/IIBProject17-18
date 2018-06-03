import glob
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import gdal
from gdalconst import *

class myAugmentation(object):
    """
    A class used to augmentate image
    Firstly, read train image and label seperately, and then merge them together for the next process
    Secondly, use keras preprocessing to augmentate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, subimg_dim, npy_path="npydata",
                 aug_train_path="aug_train", aug_label_path="aug_label"):

        """
        Using glob to get all .img_type form path
        """
        self.subimg_dim = subimg_dim
        self.npy_path = npy_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=10,
            height_shift_range=10,
            shear_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect')

    def Augmentation(self):

        print('Start augmentation')
        imgs = np.load(self.npy_path + '/imgs_train.npy')
        print('loaded imgs_train.npy')
        # imgs_train = imgs[:, :, :, :2]  # first 2 channels, shape subimg_count x h x w x 2
        labels_train = imgs[:, :, :, 2:]  # last 3 channels, shape subimg_count x h x w x 3
        index = 0
        for i in range(imgs.shape[0]):
            if (np.sum(labels_train[i, :, :, 1] + labels_train[i, :, :, 2]) > 0.5 * imgs.shape[1] *
                    imgs.shape[2]):
                img = imgs[i, :, :, :]  # shape     h x w x 5
                img = img.reshape((1,) + img.shape)  # shape 1 x h x w x 5
                savedir_t = self.aug_train_path
                if not os.path.lexists(savedir_t):
                    os.mkdir(savedir_t)
                savedir_l = self.aug_label_path
                if not os.path.lexists(savedir_l):
                    os.mkdir(savedir_l)
                self.doAugmentate(img, savedir_t, savedir_l, index)
                index += 1
        print('Augmentation complete.')

    def doAugmentate(self, img, savedir_t, savedir_l, save_prefix, batch_size=1, imgnum=5):

        # augmentate one image
        aug_array = []
        i = 1
        for batch in self.datagen.flow(img, batch_size=batch_size):
            aug_array.append(batch)
            i += 1
            if i > imgnum:
                break
        augmented_data = np.concatenate(aug_array)
        train = augmented_data[:, :, :, :2]  # first 2 channels, has shape imgnum x h x w x 2
        label = augmented_data[:, :, :, 2:]  # last 3 channels, has shape imgnum x h x w x 3
        np.save(savedir_t + '/' + str(save_prefix) + '.npy', train)
        np.save(savedir_l + '/' + str(save_prefix) + '.npy', label)


class dataProcess(object):

    def __init__(self, out_rows, out_cols, npy_path="npydata",
                 aug_train_path="aug_train", aug_label_path="aug_label",
                 img_filename="openSeaSARData.tif"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.npy_path = npy_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.img_filename = img_filename

    def create_imgs_train(self):
        if not os.path.isfile(img_filename):
            print('Please first download the file openSeaSARData.tif from the googleDriveURL')
            return -1
        if not os.path.isfile(self.npy_path + "/imgs.npy"):
            print('Please first run the Jupyter Notebook CreateArraysForMulticlassImagesAndLabels.ipynb')
            return -1
        if not os.path.isfile(self.npy_path + "/labels.npy"):
            print('Please first run the Jupyter Notebook CreateArraysForMulticlassImagesAndLabels.ipynb')
            return -1
        subimg_dim = self.out_rows

        train_list = []
        test_list = []

        ############### data from open sea ###############

        subimg_count = int(5120 / subimg_dim * 5120 / subimg_dim)
        np.random.seed(0)
        test_indices = np.random.choice(subimg_count, size=int(subimg_count * 0.2), replace=False)

        dataset = gdal.Open(self.img_filename, GA_ReadOnly)
        xSize = dataset.RasterXSize
        ySize = dataset.RasterYSize
        bandCount = dataset.RasterCount
        mean = []
        std = []
        print('TIFF image size is ', xSize, 'x', ySize, 'x', bandCount)

        for bandNumber in range(1, bandCount):
            band = dataset.GetRasterBand(bandNumber)
            bandArray = band.ReadAsArray(0, 0, 5120, 5120).astype(np.float32)
            b = bandArray.ravel()
            mean.append(b.mean(axis=0))
            std.append((b - mean[bandNumber - 1]).std(axis=0))
        mean.append(0)
        std.append(1)

        print('Saving open sea subimages -> Started')
        subimg_count = int(ySize / subimg_dim * xSize / subimg_dim)
        img_array_list = []
        test_img_array_list = []
        train_index = 0
        test_index = 0

        for yOffset in range(0, ySize, subimg_dim):
            for xOffset in range(0, xSize, subimg_dim):
                band = dataset.GetRasterBand(3)
                bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                if bandArray.sum() > 0.0:
                    _id = int(yOffset / subimg_dim * ySize / subimg_dim + xOffset / subimg_dim)
                    if _id in test_indices:
                        test_img_array = np.ndarray((1, subimg_dim, subimg_dim, 5), dtype=np.float32)
                        for bandNumber in range(1, bandCount):
                            band = dataset.GetRasterBand(bandNumber)
                            bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                            test_img_array[0, :, :, bandNumber - 1] = (bandArray - mean[bandNumber - 1]) / std[
                                bandNumber - 1]
                        band = dataset.GetRasterBand(3)
                        bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                        test_img_array[0, :, :, 4] = bandArray          # ship mask
                        test_img_array[0, :, :, 3] = 1.0 - bandArray    # sea mask
                        test_img_array[0, :, :, 2].fill(0.0)            # land mask

                        test_img_array_list.append(test_img_array)
                        test_index += 1
                    else:
                        img_array = np.ndarray((1, subimg_dim, subimg_dim, 5), dtype=np.float32)
                        for bandNumber in range(1, bandCount):
                            band = dataset.GetRasterBand(bandNumber)
                            bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                            img_array[0, :, :, bandNumber - 1] = (bandArray - mean[bandNumber - 1]) / std[
                                bandNumber - 1]
                        band = dataset.GetRasterBand(3)
                        bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                        img_array[0, :, :, 4] = bandArray
                        img_array[0, :, :, 3] = 1.0 - bandArray
                        img_array[0, :, :, 2].fill(0.0)

                        img_array_list.append(img_array)
                        train_index += 1
        img_array1 = np.concatenate(img_array_list)
        test_img_array1 = np.concatenate(test_img_array_list)
        print('Done: {}/{} subimages from OpenSea'.format(_id, subimg_count))
        print('open sea train imgs shape: ', img_array1.shape)
        print('open sea test imgs shape: ', test_img_array1.shape)
        train_list.append(img_array1)
        test_list.append(test_img_array1)

        #### data from coastal area saved as imgs.npy and labels.npy in npydata folder #######

        subimg_count = int(3072 / subimg_dim * 3072 / subimg_dim)
        np.random.seed(0)
        test_indices = np.random.choice(subimg_count, size=int(subimg_count * 0.1), replace=False)
        imgs = np.load(self.npy_path + "/imgs.npy")
        print('Loaded imgs.npy')
        labels = np.load(self.npy_path + "/labels.npy")
        print('Loaded labels.npy')
        ySize = imgs.shape[0]
        xSize = imgs.shape[1]

        for band in range(imgs.shape[2]):
            bandArray = imgs[:, :, band]
            b = bandArray.ravel()
            _mean = b.mean(axis=0)
            _std = (b - _mean).std(axis=0)
            imgs[:, :, band] = (imgs[:, :, band] - _mean) / _std

        print('Saving coastal area subimages -> Started')
        subimg_count = int(ySize / subimg_dim * xSize / subimg_dim)
        img_array_list = []
        test_img_array_list = []
        train_index = 0
        test_index = 0

        for yOffset in range(0, ySize, subimg_dim):
            for xOffset in range(0, xSize, subimg_dim):
                if labels[yOffset:yOffset + subimg_dim, xOffset:xOffset + subimg_dim,
                   0].sum() < 0.5 * subimg_dim * subimg_dim:
                    _id = int(yOffset / subimg_dim * ySize / subimg_dim + xOffset / subimg_dim)
                    if _id in test_indices:
                        test_img_array = np.ndarray((1, subimg_dim, subimg_dim, 5), dtype=np.float32)
                        for band in range(imgs.shape[2]):
                            test_img_array[0, :, :, band] = imgs[yOffset:yOffset + subimg_dim,
                                                            xOffset:xOffset + subimg_dim, band]
                        for band in range(labels.shape[2]):
                            test_img_array[0, :, :, band + imgs.shape[2]] = labels[yOffset:yOffset + subimg_dim,
                                                                            xOffset:xOffset + subimg_dim, band]
                        test_img_array_list.append(test_img_array)
                        test_index += 1
                    else:
                        img_array = np.ndarray((1, subimg_dim, subimg_dim, 5), dtype=np.float32)
                        for band in range(imgs.shape[2]):
                            img_array[0, :, :, band] = imgs[yOffset:yOffset + subimg_dim,
                                                       xOffset:xOffset + subimg_dim, band]
                        for band in range(labels.shape[2]):
                            img_array[0, :, :, band + imgs.shape[2]] = labels[yOffset:yOffset + subimg_dim,
                                                                       xOffset:xOffset + subimg_dim, band]
                        img_array_list.append(img_array)
                        train_index += 1
        img_array2 = np.concatenate(img_array_list)
        test_img_array2 = np.concatenate(test_img_array_list)
        print('coastal area train imgs shape: ', img_array2.shape, '. Done {}/{}'.format(train_index, subimg_count-len(test_indices)), )
        print('coastal area test imgs shape: ', test_img_array2.shape, '. Done {}/{}'.format(test_index, len(test_indices)),)
        train_list.append(img_array2)
        test_list.append(test_img_array2)

        train = np.concatenate(train_list)
        test = np.concatenate(test_list)
        print('final train shape: ', train.shape)
        print('final test shape: ', test.shape)
        np.save(self.npy_path + '/imgs_train.npy', train)
        np.save(self.npy_path + '/imgs_test.npy', test)
        print('Saved all subimage')

    def load_train_data(self, mask):
        # land : mask=0
        # sea  : mask=1
        # ships: mask=2
        print('-'*30)
        print('loading train images into one array...')
        print('-'*30)
        trains = glob.glob(self.aug_train_path + "/*.npy")
        labels = glob.glob(self.aug_label_path + "/*.npy")
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        aug = np.load(self.aug_train_path + "/0.npy").shape[0]
        imgs_train = np.ndarray((len(trains)*aug, self.out_rows, self.out_cols, 2), dtype=np.float32)
        labels_train = np.ndarray((len(trains)*aug, self.out_rows, self.out_cols), dtype=np.float32)
        index = 0
        for i in range(len(trains)):
            imgs_train[index:index+aug,:,:,:] = np.load(self.aug_train_path + "/" + str(i) + ".npy")     # has shape imgnum x h x w x 2
            labels_train[index:index + aug, :, :] = np.load(self.aug_label_path + "/" + str(i) + ".npy")[:, :, :, mask]  # has shape imgnum x h x w
            index += aug
        labels_train = labels_train.reshape(labels_train.shape + (1,))
        print('train images loaded.')
        print('-' * 30)
        return imgs_train, labels_train

    def load_test_data(self, mask):
        print('-'*30)
        print('loading test images...')
        print('-'*30)
        imgs_test_merged = np.load(self.npy_path+"/imgs_test.npy")
        imgs_test = imgs_test_merged[:, :, :, :2]  # first 2 channels, final shape subimg_count x h x w x 2
        labels_test = imgs_test_merged[:, :, :, mask+2]  # mask channel, shape subimg_count x h x w
        labels_test = labels_test.reshape(labels_test.shape + (1,))
        print('test images loaded.')
        print('-' * 30)
        return imgs_test, labels_test


if __name__ == "__main__":

    """
    If training One-vs-All Classifier for
        land : set mask=0
        sea  : set mask=1
        ships: set mask=2
    """
    mask = 0
    mydata = dataProcess(128, 128)
    mydata.create_imgs_train()
    aug = myAugmentation(128)
    aug.Augmentation()
    imgs_train, labels_train = mydata.load_train_data(mask)
    print('train imgs shape: ', imgs_train.shape, ', train labels shape: ', labels_train.shape)
    imgs_test, labels_test = mydata.load_test_data(mask)
    print('test imgs shape: ', imgs_test.shape, ', test labels shape: ', labels_test.shape)