###### *more info about the project in report.html*

# Requirements

1. Python3
2. Install all python libraries in requirements.txt
3. Matlab

# How to reproduce our result
```shell
python3 HDR.py
matlab tonemapping.m
```
## Usage: HDR.py
```
usage: HDR.py [-h] [-d DIRECTORY] [-p PREFIX] [-n NUMBER] [-i INDEX]
              [-s SUFFIX] [-f FILENAME] [-a ARRAY]

optional arguments:
  -h, --help    show this help message and exit
  -d DIRECTORY  directory of the input images
  -p PREFIX     prefix of the images
  -n NUMBER     total number of images
  -i INDEX      index of first image
  -s SUFFIX     file type of image
  -f FILENAME   filename of the output numpy array
  -a ARRAY      exist saved np files for images and enter its prefix
```
