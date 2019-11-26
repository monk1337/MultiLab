# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='multilab',
    version='0.1.0',
    description='Framework for multi-label classification',
    long_description=readme,
    author='Human',
    author_email='checkmate',
    url='setting',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)