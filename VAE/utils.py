import numpy as np

def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    imgs = np.zeros(h*size[0], w*size[1])
    for idx, img in enumerate(imgs):
        i = int(idx / size[0])
        j = idx % size[1]
        imgs[i*h:i*h+h, j*w:j*w+w] = img

    return imgs

