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
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]              # shape     h x w x 3
            img = img.reshape((1,) + img.shape) # shape 1 x h x w x 3
            if not os.path.lexists(savedir_t):
                os.mkdir(savedir_t)
            savedir_l = self.aug_label_path
            if not os.path.lexists(savedir_l):
                os.mkdir(savedir_l)
            self.doAugmentate(img, savedir_t, savedir_l, i)
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
        label = augmented_data[:, :, :, 2]  # last channel
        label = label.reshape(label.shape + (1,))  # has shape imgnum x h x w x 1
        np.save(savedir_t + '/' + str(save_prefix) + '.npy', train)
        np.save(savedir_l + '/' + str(save_prefix) + '.npy', label)


class dataProcess(object):

    def __init__(self, out_rows, out_cols, npy_path="npydata",
                 aug_train_path="aug_train", aug_label_path="aug_label"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.npy_path = npy_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path

    def create_imgs_train(self):
        img_filename = "LabelledSARdata.tif"
        if not os.path.isfile(img_filename):
            print('Please first download the file LabelledSARdata.tif from the googleDriveURL')
            return -1
        target_file_path = self.npy_path
        subimg_dim = self.out_rows
        subimg_count = int(5120 / subimg_dim * 5120 / subimg_dim)
        test_indices = np.random.choice(subimg_count, size=int(subimg_count * 0.1), replace=False)

        dataset = gdal.Open(img_filename, GA_ReadOnly)
        xSize = dataset.RasterXSize
        ySize = dataset.RasterYSize
        bandCount = dataset.RasterCount
        mean = []
        std = []
        print('Input image size is ', xSize, 'x', ySize, 'x', bandCount)

        for bandNumber in range(1, bandCount):
            band = dataset.GetRasterBand(bandNumber)
            bandArray = band.ReadAsArray(0, 0, 5120, 5120).astype(np.float32)
            b = bandArray.ravel()
            mean.append(b.mean(axis=0))
            std.append((b - mean[bandNumber - 1]).std(axis=0))
        mean.append(0)
        std.append(1)

        print('Saving subimages -> Started')
        subimg_count = int(ySize / subimg_dim * xSize / subimg_dim)
        img_array = np.ndarray((subimg_count, subimg_dim, subimg_dim, 3), dtype=np.float32)
        test_img_array = np.ndarray((len(test_indices), subimg_dim, subimg_dim, 3), dtype=np.float32)
        index = 0

        for yOffset in range(0, ySize, subimg_dim):
            for xOffset in range(0, xSize, subimg_dim):
                _id = int(yOffset / subimg_dim * ySize / subimg_dim + xOffset / subimg_dim)
                if _id in test_indices:
                    for bandNumber in range(1, bandCount + 1):
                        band = dataset.GetRasterBand(bandNumber)
                        bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                        img_array[_id, :, :, bandNumber - 1] = (bandArray-mean[bandNumber-1])/std[bandNumber-1]
                        test_img_array[index, :, :, bandNumber - 1] = (bandArray-mean[bandNumber-1])/std[bandNumber-1]
                    index += 1
                else:
                    for bandNumber in range(1, bandCount + 1):
                        band = dataset.GetRasterBand(bandNumber)
                        bandArray = band.ReadAsArray(xOffset, yOffset, subimg_dim, subimg_dim).astype(np.float32)
                        img_array[_id, :, :, bandNumber - 1] = (bandArray-mean[bandNumber-1])/std[bandNumber-1]
                if _id % (subimg_count / 10) == 0:
                    print('Done: {}/{} subimages'.format(_id, subimg_count))
        np.save(target_file_path + '/imgs_train.npy', img_array)
        np.save(target_file_path + '/imgs_test.npy', test_img_array)
        print('Saved: {}/{} train subimages'.format(_id + 1, subimg_count) + 'in {}/imgs_train.npy'.format(
            target_file_path))
        print('Saved: {}/{} test subimages'.format(index, len(test_indices)) + 'in {}/imgs_test.npy'.format(
            target_file_path))

    def load_train_data(self):
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
        labels_train = np.ndarray((len(trains)*aug, self.out_rows, self.out_cols, 1), dtype=np.float32)
        index = 0
        for i in range(len(trains)):
            imgs_train[index:index+aug,:,:,:] = np.load(self.aug_train_path + "/" + str(i) + ".npy")     # has shape imgnum x h x w x 2
            labels_train[index:index+aug,:,:,:] = np.load(self.aug_label_path + "/" + str(i) + ".npy")   # has shape imgnum x h x w x 1
            index += aug
        print('train images loaded.')
        print('-' * 30)
        return imgs_train, labels_train

    def load_test_data(self, mask):
        print('-'*30)
        print('loading test images...')
        print('-'*30)
        imgs_test_merged = np.load(self.npy_path+"/imgs_test.npy")
        imgs_test = imgs_test_merged[:, :, :, :2]  # first 2 channels, final shape subimg_count x h x w x 2
        labels_test = imgs_test_merged[:, :, :, mask+2]
        labels_test = labels_test.reshape(labels_test.shape + (1,))  # final shape subimg_count x h x w x 1
        print('test images loaded.')
        print('-' * 30)
        return imgs_test, labels_test


if __name__ == "__main__":

    mydata = dataProcess(128, 128)
    mydata.create_imgs_train()
    aug = myAugmentation(128)
    aug.Augmentation()
    imgs_train, labels_train = mydata.load_train_data()
    print('train imgs shape: ', imgs_train.shape, ', train labels shape: ', labels_train.shape)
    imgs_test, labels_test = mydata.load_test_data()
    print('test imgs shape: ', imgs_test.shape, 'test labels shape:', labels_test)

