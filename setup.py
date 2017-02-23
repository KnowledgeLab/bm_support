import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bm_support",
    version="0.0.1",
    author="Alexander Belikov",
    author_email="abelikov@gmail.com",
    description="tools for big mechanism project",
    license="BSD",
    keywords="pandas",
    # url="git@github.com:alexander-belikov/datahelpers.git",
    packages=['bm_support'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 0 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        'pandas', 'setuptools',
        'datahelpers', 'scipy',
        'pymc3'

    ],
)
