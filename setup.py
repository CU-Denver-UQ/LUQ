#!/usr/bin/env python

from distutils.core import setup

setup(name='luq',
      version='1.0',
      description='Learning Uncertain Quantities',
      author='Steven Mattis',
      author_email='steve.a.mattis@gmail.com',
      url='https://github.com/CU-Denver-UQ/LUQ',
      packages=['luq'],
      install_requires=['matplotlib', 'scipy',
                        'numpy', 'scikit-learn']
      )

