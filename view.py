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
    description='Visualize TARLOADER archive content.',
    formatter_class=RawTextHelpFormatter)
ap.add_argument('--image-shape', type=int, required=False,
    nargs=2, metavar=('<width>','<height>'), default=[224,224],
    help='Image shape. Default: [224,224].')

ImageArchive_add_parser_options(ap)
args = ap.parse_args()

dataset = ImageArchive (
    apath = args.tarloader_archive,
    ipath = args.tarloader_index_file,
    root = args.tarloader_archive_root_directory,
    image_index = args.tarloader_image_file,
    batch_size = args.tarloader_batch_size,
    drop_last = args.tarloader_drop_last_batch,
    shuffle = args.tarloader_shuffle,
    overwrite_index = args.tarloader_index_file_overwrite,
    data_in_memory = args.tarloader_keep_in_memory,
    open_after_fork = args.tarloader_enable_multiprocess
)

ncols = int(sqrt(args.tarloader_batch_size))
nrows = int(ceil(args.tarloader_batch_size/ncols))

def img_generator():
    for imgs, labels in dataset:
        res = [img.resize(args.image_shape) for img in imgs]
        res = [add_textbox(img,f'Class {label[0]}') for img,label in zip(res,labels)]
        yield res

visualizer = ImageVisualizer(
    img_generator(),
    nrows = nrows,
    ncols = ncols,
    wscreen = 1500,
    hscreen = 900,
    focus = None,
    use_col_textboxes = False,
    use_row_textboxes = False,
    verbose = False
)
