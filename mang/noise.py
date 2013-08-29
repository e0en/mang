#!/usr/bin/python
#coding=utf-8

"""
mang.noise
~~~~~~~~~~

Stochastic noises to be added to inputs of neural networks.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import cudamat.gnumpy as gnp


def gaussian(x, option):
    """Add gaussian noise with a fixed variance and return the result."""
    return x + option["level"] * gnp.randn(*x.shape)



def salt(x, option):
    """Randomly replace elements of inputs by 1."""
    mask = gnp.rand(*x.shape) < option["level"]
    return (1. - mask) * x + mask


def pepper(x, option):
    """Randomly replace elements of inputs by 0."""
    mask = gnp.rand(*x.shape) > option["level"]
    return mask * x


noise_table = {
    "gaussian": gaussian,
    "salt": salt,
    "pepper": pepper,
    }
