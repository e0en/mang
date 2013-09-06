#!/usr/bin/python
#coding=utf-8

"""
mang.cpu_conv
~~~~~~~~~~~~~

CPU-version of convolution routines of cudamat-conv.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import cudamat as cm
from cudamat import cudamat_conv as cm_conv
import numpy as np


def conv_up(x, filters, n_channel, padding, stride):
    """Apply convolution on x using filters.

    Arguments:
        x: input images, (n_image, x_size ** 2 * n_channel)
        filters: convolution filters,
                 (n_filter, filter_size ** 2 * n_channel)
        n_channel: number of colors in images
        padding: size of zero-padding on output
        n_filter: number of filters
        stride: distance between convolution locations in pixels

    Returns:
        o: convoluted images

    Raises:
        Nothing.

    """
    n_image = x.shape[0]
    x_size = int(np.sqrt(x.shape[1] / n_channel))
    n_filter = filters.shape[0]
    filter_size = int(np.sqrt(filters.shape[1] / n_channel))
    o_size = (x_size + 2 * padding - filter_size) / stride + 1
    batch_size = 128
    o = np.zeros((n_image, o_size ** 2 * n_filter))
    w_cm = cm.CUDAMatrix(filters)
    n_batch = int(np.ceil(1. * n_image / batch_size))
    batch = cm.empty((batch_size, x.shape[1]))
    o_cm = cm.empty((batch_size, o.shape[1]))
    for i_batch in xrange(n_batch):
        i1 = i_batch * batch_size
        i2 = min(i1 + batch_size, n_image)
        n_sample = i2 - i1
        if n_sample == batch_size:
            batch.overwrite(x[i1:i2])
        else:
            tmp = np.zeros((batch_size, x.shape[1]))
            tmp[:n_sample] = x[i1:i2]
            batch.overwrite(tmp)
        cm_conv.convUp(batch, w_cm, o_cm, o_size, padding, stride, n_channel)
        o[i1:i2] = o_cm.asarray()[:n_sample]
    batch.free_device_memory()
    o_cm.free_device_memory()
    w_cm.free_device_memory()
    del batch, o_cm, w_cm
    return o


def max_pool(x, n_channel, ratio, stride):
    """Apply convolution on x using filters.

    Arguments:
        x: input images, (n_image, x_size ** 2 * n_channel)
        n_channel: number of colors in images
        ratio: the width of pooling area
        stride: distance between convolution locations in pixels

    Returns:
        o: subsampled images

    Raises:
        Nothing.

    """

    n_image = x.shape[0]
    x_size = int(np.sqrt(x.shape[1] / n_channel))
    o_size = (x_size - ratio) / stride + 1
    o = np.zeros((n_image, o_size ** 2 * n_channel))

    batch_size = 128
    batch = cm.empty((batch_size, x.shape[1]))
    o_cm = cm.empty((batch_size, o.shape[1]))

    n_batch = int(np.ceil(1. * n_image / batch_size))
    for i_batch in xrange(n_batch):
        i1 = i_batch * batch_size
        i2 = min(i1 + batch_size, n_image)
        n_sample = i2 - i1
        if n_sample == batch_size:
            batch.overwrite(x[i1:i2])
        else:
            tmp = np.zeros((batch_size, x.shape[1]))
            tmp[:n_sample] = x[i1:i2]
            batch.overwrite(tmp)
        cm_conv.MaxPool(batch, o_cm, n_channel, ratio, 0, stride, o_size)
        o[i1:i2] = o_cm.asarray()[:n_sample]
    batch.free_device_memory()
    o_cm.free_device_memory()
    del batch, o_cm
    return o


def response_normalization(x, n_channel, size, add_scale, pow_scale):
    """Run response normalization on inputs.
    o = x * (1 + add_scale * cov)^(-pow_scale)
    cov = sum_(region) x^2
    region size = (size, size)

    Arguments:
        x: input images, (n_image, x_size ** 2 * n_channel)
        n_channel: number of colors in images
        size: the width of response normalization area

    Returns:
        o: response-normalized images

    Raises:
        Nothing.

    """

    n_image = x.shape[0]
    x_size = int(np.sqrt(x.shape[1] / n_channel))
    o = np.zeros((n_image, x_size ** 2 * n_channel))
    batch_size = 128

    batch = cm.empty((batch_size, x.shape[1]))
    o_cm = cm.empty((batch_size, o.shape[1]))
    cov_cm = cm.empty((batch_size, x.shape[1]))

    n_batch = int(np.ceil(1. * n_image / batch_size))
    for i_batch in xrange(n_batch):
        i1 = i_batch * batch_size
        i2 = min(i1 + batch_size, n_image)
        n_sample = i2 - i1
        if n_sample == batch_size:
            batch.overwrite(x[i1:i2])
        else:
            tmp = np.zeros((batch_size, x.shape[1]))
            tmp[:n_sample] = x[i1:i2]
            batch.overwrite(tmp)
        cm_conv.ResponseNorm(batch, cov_cm, o_cm, n_channel, size,
                             add_scale, pow_scale)
        o[i1:i2] = o_cm.asarray()[:n_sample]
    batch.free_device_memory()
    o_cm.free_device_memory()
    cov_cm.free_device_memory()
    del batch, o_cm, cov_cm
    return o


