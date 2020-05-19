import os
import scipy
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt 

def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims, name=name)


def softmax(logits, axis=None, name=None):
    try:
        return tf.nn.softmax(logits, axis=axis, name=name)
    except:
        return tf.nn.softmax(logits, dim=axis, name=name)
 

def euclidean_norm(input, axis=2, keepdims=True, epsilon=True):
    if epsilon:
        norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims) + 1e-9)
    else:
        norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims))

    return(norm)

# -------------------------------------------------------------
def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


def get_transformation_matrix_shape(in_pose_shape, out_pose_shape):
    return([out_pose_shape[0], in_pose_shape[0]])
