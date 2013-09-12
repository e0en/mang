#!/usr/bin/python
#coding=utf-8

"""
This is a setup script for mang.

.. moduleauthor:: Yoonseop Kang <e0engoon@gmail.com>

"""
import os
from setuptools import setup
from setuptools.command.install import install


class MyInstall(install):
    def run(self):
        os.system("cd mang/cudamat; rm *.so; make")
        os.system("cd ../../")
        install.run(self)

CUDAMAT_FILES = [
    "*.cu", "*.cuh", "*.so", "*.txt", "run_on_me_or_pid_quit", "Makefile"]

setup(
    name="mang",
    version="0.1.1",
    description="Another neural network library for python based on cudamat",

    install_requires=["numpy>=1.6.1", "msgpack-python", "msgpack_numpy"],

    packages=["mang", "mang.node", "mang.edge", "mang.cudamat"],
    package_data={"mang.cudamat": CUDAMAT_FILES, },
    cmdclass={"install": MyInstall},

    author="Yoonseop Kang",
    author_email="e0engoon@gmail.com",
    url="http://github.com/e0en/mang",
    keywords="neural network, artificial intelligence, machine learning",
    zip_safe=False)
