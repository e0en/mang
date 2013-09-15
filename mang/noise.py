#!/usr/bin/python
#coding=utf-8

"""
mang.noise
~~~~~~~~~~

Stochastic noises to be added to inputs of neural networks.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import cudamat as cm


def gaussian(x, option):
    """Add gaussian noise with a fixed variance and return the result."""

    x.sample_gaussian(option)


def salt(x, option):
    """Randomly replace elements of inputs by 1."""

    x.dropout(option, 1.)


def pepper(x, option):
    """Randomly replace elements of inputs by 0."""

    x.dropout(option, 0.)


NOISE_TABLE = {
    "gaussian": gaussian,
    "salt": salt,
    "pepper": pepper,
    }
