#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="video_processor",
    version="0.1.0",
    description="Video processing tool with YOLO object detection",
    author="Jean-Paul M. Chia",
    author_email="jpmchia@gmail.com",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=11.0.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "moviepy>=1.0.3",
        "ipywidgets>=7.6.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        "console_scripts": [
            "video-processor=video_processor.__main__:main_cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
