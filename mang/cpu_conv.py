#!/usr/bin/python
#coding=utf-8

"""
mang.cpu_conv
~~~~~~~~~~~~~

CPU-version of convolution routines of cudamat-conv.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import os
import ctypes as ct

import numpy as np

current_dir = os.path.realpath(__file__)
dll_path = os.path.join(current_dir, "libcpuconv.so")
_conv = ct.CDLL(dll_path)


def conv_up(x, filters, n_channel, stride):
    """Apply convolution on x using filters.

    Arguments:
        x: input images, (n_image, x_size ** 2 * n_channel)
        filters: convolution filters,
                 (n_filter, filter_size ** 2 * n_channel)
        n_channel: number of colors in images
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
    o_size = (x_size - filter_size) / stride + 1

    """
    x = np.array(x, dtype=np.float32)
    o = np.zeros((n_image, o_size ** 2 * n_filter), dtype=np.float32)
    _conv.conv_up(x.ctypes.data_as(ct.POINTER),
                  o.ctypes.data_as(ct.POINTER),
                  x_size, n_filter, filter_size, o_size)
    return o
    """

    for i_image in xrange(n_image):
        img_x = x[i_image].reshape((x_size, x_size, n_channel), order="F")
        img_o = o[i_image].reshape((o_size, o_size, n_filter), order="F")
        for i_filter in xrange(n_filter):
            img_filter = \
                filters[i_filter].reshape((filter_size, filter_size,
                                          n_channel), order="F")
            for i_x in xrange(o_size):
                for i_y in xrange(o_size):
                    i1 = i_x * stride
                    i2 = i1 + filter_size
                    j1 = i_y * stride
                    j2 = j1 + filter_size
                    img_o[i_x, i_y, i_filter] = \
                        (img_x[i1:i2, j1:j2, :] * img_filter).sum()
        o[i_image] = img_o.reshape((o_size * o_size * n_filter), order="F")
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

    """
    x = np.array(x, dtype=np.float32)
    o = np.zeros((n_image, o_size ** 2 * n_channel), dtype=np.float32)
    _conv.max_pool(x.ctypes.data_as(ct.POINTER),
                  o.ctypes.data_as(ct.POINTER),
                  x_size, n_channel, ratio, stride)
    return o
    """

    for i_image in xrange(n_image):
        img_x = x[i_image].reshape((x_size, x_size, n_channel), order="F")
        img_o = o[i_image].reshape((o_size, o_size, n_channel))
        for i_channel in xrange(n_channel):
            for i_x in xrange(o_size):
                for i_y in xrange(o_size):
                    i1 = i_x * stride
                    i2 = i1 + ratio
                    j1 = i_y * stride
                    j2 = j1 + ratio
                    tmp = img_x[i1:i2, j1:j2, i_channel].ravel()
                    img_o[i_x, i_y, i_channel] = tmp.max()
        o[i_image] = img_o.reshape((n_channel * o_size * o_size, ), order="F")
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
    o = np.array(x, order="F")
    p = np.array(x, order="F")

    """
    x = np.array(x, dtype=np.float32)
    o = np.zeros((n_image, x_size ** 2 * n_channel), dtype=np.float32)
    _conv.response_normalization(x.ctypes.data_as(ct.POINTER),
                                 o.ctypes.data_as(ct.POINTER),
                                 x_size, n_channel, ratio, stride)
    return o
    """


    for i_image in xrange(n_image):
        img_x = x[i_image].reshape((x_size, x_size, n_channel), order="F")
        img_o = o[i_image].reshape((x_size, x_size, n_channel), order="F")
        img_p = p[i_image].reshape((x_size, x_size, n_channel), order="F")
        for i_channel in xrange(n_channel):
            for i_x in xrange(x_size):
                for i_y in xrange(x_size):
                    i0 = i_x - size / 2
                    j0 = i_y - size / 2
                    i1 = max(0, i0)
                    j1 = max(0, j0)
                    i2 = min(x_size, i0 + size)
                    j2 = min(x_size, j0 + size)
                    prod = img_x[i1:i2, j1:j2, :]
                    prod = 1. + add_scale * (prod * prod).sum(0).sum(0)
                    img_p[i_x, i_y, :] = prod
                    img_o[i_x, i_y, :] = \
                        img_x[i_x, i_y, :] * np.power(prod, -pow_scale)

        o[i_image] = img_o.reshape((n_channel * x_size * x_size, ), order="F")
        p[i_image] = img_p.reshape((n_channel * x_size * x_size, ), order="F")
    return (o, p)
