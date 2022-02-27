#!/usr/bin/env python3
from tarloader import ImageArchive, ImageArchive_add_parser_options
from image_visualizer import ImageVisualizer
from argparse import RawTextHelpFormatter
from math import sqrt,ceil
from typing import Optional
import os
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import argparse

def add_textbox (
    img: Image.Image,
    content: str,
    font_size: Optional[int] = 20,
) -> Image.Image:
    """ Add textbox to image
    Args:
        img (Image.Image): Source image
        content (str): Text content
        font_size (optional,int): Font size
    """
    w = img.width
    h = font_size+4
    y = img.height-h
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, y, w, y+h), fill='white')
    # Get path to font
    path = os.path.dirname(os.path.abspath(__file__))+'/fonts/arial.ttf'
    font = ImageFont.truetype(path, font_size)
    draw.text((w/2, y+h/2), content, fill='black', anchor='mm', font=font)
    return img


# Parse command line
ap = argparse.ArgumentParser(
    description='Visualize TARLOADER archive(s) content.',
    formatter_class=RawTextHelpFormatter)
ap.add_argument('--image-shape', type=int, required=False,
    nargs=2, metavar=('<width>','<height>'), default=[224,224],
    help='Image shape. Default: [224,224].')

ImageArchive_add_parser_options(ap)
args = ap.parse_args()

if len(args.tarloader_archive) not in [1, len(args.tarloader_index_file)]:
    ap.error('Mismatching number of TAR archives and index files.')
if len(args.tarloader_archive) == 1:
    # Broadcast archive file
    args.tarloader_archive = [args.tarloader_archive[0]
            for _ in range(len(args.tarloader_index_file))]

datasets = [ImageArchive (
    apath = archive,
    ipath = index,
    batch_size = args.tarloader_batch_size,
    drop_last = args.tarloader_drop_last_batch,
    shuffle = args.tarloader_shuffle,
    data_in_memory = args.tarloader_keep_in_memory,
    open_after_fork = args.tarloader_enable_multiprocess
) for archive, index in zip(args.tarloader_archive,args.tarloader_index_file)]

ncols = args.tarloader_batch_size
nrows = len(datasets)

# Image generator for each dataset
def dataset_img_generator(dataset):
    for imgs, labels in dataset:
        res = [img.resize(args.image_shape) for img in imgs]
        res = [add_textbox(img,f'Class {label[0]}') for img,label in zip(res,labels)]
        yield res

# Aggregator
def aggregator ():
    generators = [dataset_img_generator(d) for d in datasets]
    end = False
    while not(end):
        imgs = []
        for g in generators:
            try :
                imgs += next(g)
            except StopIteration:
                end = True
        if end: raise StopIteration()
        yield imgs

visualizer = ImageVisualizer(
    aggregator(),
    nrows = nrows,
    ncols = ncols,
    wscreen = 1500,
    hscreen = 900,
    focus = None,
    use_col_textboxes = False,
    use_row_textboxes = False,
    verbose = False
)
