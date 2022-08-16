#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="anbinx-dl-env",
    version="0.0.1",
    description="My deeplearning environment",
    author="anbinx",
    author_email="",
    url="https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
