import cPickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import random


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def wpickle(diction, filename):
    with open(filename, mode='wb') as f:
        pickle.dump(diction, f, protocol=2)




def load_dataset(path):
    dataset = unpickle(path)
    return dataset




# print(test_set[1].shape[0])


def create_noise(test_set, percentage):
    num = test_set[1].shape[0]*percentage/100.0
    num = int(round(num))
# print(num)

    for i in range(0, test_set.shape[0]):
        index = []
        while len(index) < num:
            ind = random.randint(0, test_set.shape[1]-1)
            if ind not in index:
                index.append(ind)
    # print(len(index))

        for ind in index:
            pix = random.randint(0, 255)
            test_set[i][ind] = pix
    return test_set


#create_noise(test_set, 5.0)

filelist = ["./cifar-10-python/noise_5/data_batch_1",
            "./cifar-10-python/noise_5/data_batch_2",
            "./cifar-10-python/noise_5/data_batch_3",
            "./cifar-10-python/noise_5/data_batch_4",
            "./cifar-10-python/noise_5/data_batch_5",
            "./cifar-10-python/noise_5/test_batch",
            "./cifar-10-python/noise_10/data_batch_1",
            "./cifar-10-python/noise_10/data_batch_2",
            "./cifar-10-python/noise_10/data_batch_3",
            "./cifar-10-python/noise_10/data_batch_4",
            "./cifar-10-python/noise_10/data_batch_5",
            "./cifar-10-python/noise_10/test_batch",
            "./cifar-10-python/noise_15/data_batch_1",
            "./cifar-10-python/noise_15/data_batch_2",
            "./cifar-10-python/noise_15/data_batch_3",
            "./cifar-10-python/noise_15/data_batch_4",
            "./cifar-10-python/noise_15/data_batch_5",
            "./cifar-10-python/noise_15/test_batch",
            "./cifar-10-python/noise_20/data_batch_1",
            "./cifar-10-python/noise_20/data_batch_2",
            "./cifar-10-python/noise_20/data_batch_3",
            "./cifar-10-python/noise_20/data_batch_4",
            "./cifar-10-python/noise_20/data_batch_5",
            "./cifar-10-python/noise_20/test_batch",
            "./cifar-10-python/noise_25/data_batch_1",
            "./cifar-10-python/noise_25/data_batch_2",
            "./cifar-10-python/noise_25/data_batch_3",
            "./cifar-10-python/noise_25/data_batch_4",
            "./cifar-10-python/noise_25/data_batch_5",
            "./cifar-10-python/noise_25/test_batch", ]
for i, filename in enumerate(filelist):
    if i % 6 == 0:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/data_batch_1")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
    elif i % 6 == 1:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/data_batch_2")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
    elif i % 6 == 2:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/data_batch_3")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
    elif i % 6 == 3:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/data_batch_4")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
    elif i % 6 == 4:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/data_batch_5")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
    elif i % 6 == 5:
        data_set = load_dataset(
            "./cifar-10-python/cifar-10-batches-py/test_batch")
        test_set = data_set["data"]
        if i // 6 == 0:
            data_set["data"] = create_noise(test_set, 5.0)
            wpickle(data_set, filename)
        elif i // 6 == 1:
            data_set["data"] = create_noise(test_set, 10.0)
            wpickle(data_set, filename)
        elif i // 6 == 2:
            data_set["data"] = create_noise(test_set, 15.0)
            wpickle(data_set, filename)
        elif i // 6 == 3:
            data_set["data"] = create_noise(test_set, 20.0)
            wpickle(data_set, filename)
        elif i // 6 == 4:
            data_set["data"] = create_noise(test_set, 25.0)
            wpickle(data_set, filename)
