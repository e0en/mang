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
    version="0.1",
    description="Another neural network library for python based on cudamat",
    url="http://github.com/e0en/mang",
    author="Yoonseop Kang",
    author_email="e0engoon@gmail.com",
    packages=["mang", "mang.node", "mang.edge", "mang.cudamat"],
    package_data={
        "mang.cudamat": CUDAMAT_FILES,
        },
    cmdclass={"install": MyInstall},
    zip_safe=False)
