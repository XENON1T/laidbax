#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

setup(name='wimpy',
      version='0.2.0',
      description='Statistical XENON1T Analysis for the lazy analyst.',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/XENON1T/laidbax',
      packages=['laidbax', 'laidbax.data'],
      package_dir={'laidbax': 'laidbax'},
      install_requires=requirements,
      license="MIT",
      zip_safe=False,
      keywords='laidbax',
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
)