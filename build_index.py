#!/usr/bin/env python3
from tarloader import ImageArchive,ImageArchive_add_parser_options
from argparse import RawTextHelpFormatter
import argparse

ap = argparse.ArgumentParser(
	description='Build index for tarloader.',
	formatter_class=RawTextHelpFormatter)
ImageArchive_add_parser_options(ap,init=True)
args = ap.parse_args()

if len(args.tarloader_archive) > 1 or len(args.tarloader_index_file) > 1:
    ap.error('Only one archive allowed during building of index.')
args.tarloader_archive = args.tarloader_archive[0]
args.tarloader_index_file = args.tarloader_index_file[0]

archive = ImageArchive(
	apath = args.tarloader_archive,
	ipath = args.tarloader_index_file,
	root = args.tarloader_archive_root_directory,
	image_index = args.tarloader_image_file,
	image_label = args.tarloader_label_file,
	image_label_preprocessing = None,
	split_mask = None,
	split_val = 0,
	transform = None,
	index_transform = None,
	target_transform = None,
	batch_size = 1,
	drop_last = False,
	loader = None,
	is_valid_file  = None,
	data_in_memory = False,
	open_after_fork =  False,
	overwrite_index = args.tarloader_index_file_overwrite,
)

