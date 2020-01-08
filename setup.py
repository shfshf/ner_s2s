import os

import setuptools
from setuptools import setup

install_requires = [
    "numpy",
    "keras",
    "tokenizer_tools",
    "flask",
    "flask-cors",
    "ioflow",
    "tf-crf-layer",
    "tf-attention-layer",
    "tensorflow>=1.15.0,<2.0.0",
    "deliverable-model",
    "gunicorn",
    "micro_toolkit",
    "mlflow==1.3.0"
]


setup(
    name=os.getenv("_PKG_NAME", "ner_s2s"),  # _PKG_NAME will be used in Makefile for dev release
    version="0.0.3",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/shfshf/ner_s2s",
    license="Apache 2.0",
    author="Hanfeng Song",
    author_email="1316478299@qq.com",
    description="ner_s2s",
    install_requires=install_requires,
)
