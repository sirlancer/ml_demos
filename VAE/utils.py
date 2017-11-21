import numpy as np
import tensorflow as tf
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros([h*size[0], w*size[1]])
    for idx, img in enumerate(images):
        i = int(idx / size[0])
        j = idx % size[1]
        imgs[int(i*h):int(i*h+h), int(j*w):int(j*w+w)] = img

    return imgs

