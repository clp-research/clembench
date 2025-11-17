"""
This is a setup.py file for the clembench package.

This package is needed in order to make imports work in this repo.
"""

from setuptools import find_packages, setup

setup(
    name="clembench",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "clemcore",
    ],
)
