#!/usr/bin/python
#coding=utf-8

"""
mang.noise
~~~~~~~~~~

Stochastic noises to be added to inputs of neural networks.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import cudamat as cm


def gaussian(x, option, result):
    """Add gaussian noise with a fixed variance and return the result."""
    result.fill_with_randn()
    result.mult(option["level"])
    result.add(x)


def salt(x, option, result):
    """Randomly replace elements of inputs by 1."""
    result.assign(x)
    result.dropout(option["level"], 1.)


def pepper(x, option, result):
    """Randomly replace elements of inputs by 0."""
    result.assign(x)
    result.dropout(option["level"], 0.)


noise_table = {
    "gaussian": gaussian,
    "salt": salt,
    "pepper": pepper,
    }
