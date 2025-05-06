import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import time

train_path = './b'
val_path = './b'
test_path = './b'


def ImagePreProcessing(image_patha, image_pathb, imsize):
    img1 = cv2.imread(image_patha, 0)
    img1 = cv2.resize(img1, imsize)
    img2 = cv2.imread(image_pathb, 0)
    img2 = cv2.resize(img2, imsize)

    training_image = np.dstack((img1, img2))
    print(len(training_image))
    return training_image

def savedata(source_path, new_path, rho, patch_size, imsize, data_size):
    lst = os.listdir(source_path + '/')
    filenames = [os.path.join(source_path, l) for l in lst if l[-3:] == 'jpg']
    print("Generate {} {} files from {} raw data...".format(data_size, new_path, len(filenames)))
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for i in range(len(filenames)):
        print(filenames[i])

        path1,tmp = filenames[i].split('\\')
        path2,_ = tmp.split('.')
        out = path2.find('_')
        if out < 0:
            image_path0 = filenames[i]
            image_path1 = path1+'//' + path2 + '_1.jpg'
            image_path2 = path1+'//' + path2 + '_2.jpg'
            image_path3 = path1+'//' + path2 + '_3.jpg'
            image_path4 = path1+'//' + path2 + '_4.jpg'
            image_path5 = path1+'//' + path2 + '_5.jpg'
            print(image_path0)
            print(image_path1)

            np.save(new_path + '/' + path2 + '_10' , ImagePreProcessing(image_path0,image_path1, imsize))
            np.save(new_path + '/' + path2 + '_11', ImagePreProcessing(image_path1,image_path2, imsize))
            np.save(new_path + '/' + path2 + '_12', ImagePreProcessing(image_path2,image_path3, imsize))
            np.save(new_path + '/' + path2 + '_13', ImagePreProcessing(image_path3,image_path4, imsize))
            np.save(new_path + '/' + path2 + '_14', ImagePreProcessing(image_path4,image_path5, imsize))
            np.save(new_path + '/' + path2 + '_20' , ImagePreProcessing(image_path5,image_path4, imsize))
            np.save(new_path + '/' + path2 + '_21', ImagePreProcessing(image_path4,image_path3, imsize))
            np.save(new_path + '/' + path2 + '_22', ImagePreProcessing(image_path3,image_path2, imsize))
            np.save(new_path + '/' + path2 + '_23', ImagePreProcessing(image_path2,image_path1, imsize))
            np.save(new_path + '/' + path2 + '_24', ImagePreProcessing(image_path1,image_path0, imsize))

if __name__ == "__main__":
    start = time.time()
    rho = 32
    patch_size = 128
    imsize = (128, 128)
    savedata(train_path, './b/training/', rho, patch_size, imsize, data_size=50000)
    savedata(val_path, './b/validation/', rho, patch_size, imsize, data_size=5000)
    savedata(test_path, './b/testing/', rho, patch_size, imsize, data_size=5000)
    elapsed_time = time.time() - start
    print("Generate dataset in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))