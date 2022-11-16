import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


extras = {
    'test': [
        'filelock',
        'pytest',
        'pytest-forked',
    ],
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

setup(name='RL Nuts and Bolts',
      packages=[
          'rlkits',
          'rlalgos' 
          #package for package in find_packages()
          #      if package.startswith('')
      ],
      install_requires=[
          'gym==0.25.1',
          'ipdb'
      ],
      extras_require=extras,
      description='Components for building awesome RL algorithms',
      author='Hongshan Li',
      author_email='lihongshan8128@gmail.com',
      version='0.0.0')
