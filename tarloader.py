import os
import tarfile
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path
from PIL import Image
from io import BytesIO

class ImgInfo(object):
	"""
	Class for storing all informations regarding an image in a TAR archive.
	Some fields are directly extracted from a TarInfo object while with some additional
	information is related to database management (image index, ...)
	"""
	def __init__(self,
		tarinfo : Optional[tarfile.TarInfo] = None,
		index   : Optional[int] = 0
	):
		if tarinfo is not None:
			self.name        = tarinfo.name
			self.offset_data = tarinfo.offset_data
			self.size        = tarinfo.size
		else :
			self.name        = ""
			self.offset_data = 0
			self.size        = 0
		# Other information
		self.index = index

def sort_img (
	tar_infos: List[tarfile.TarInfo],
	fp: BinaryIO,
	decode: Optional[bool]=False
) -> List[ImgInfo]:
	""" Sort images according to index file, assuming the following content:
		0 path/to/image/0
		1 path/to/image/1
		...
		N path/to/image/N
		Note: Indices are not necessarily ordered
	Args:
		tar_infos (list): List of TarInfo for all images
		fp (BinaryIO): File pointer to index file
		decode (bool, optional): Decode index file content
	"""
	lines = fp.read().splitlines()
	if decode :
		# Decode index file
		lines = [l.decode('utf-8') for l in lines]

	img_infos = []
	for t in tar_infos:
		idx = -1
		for l in lines:
			if t.name == l.split()[1]:
				idx = int(l.split()[0])
				break
		if idx >= 0:
			img_infos.append(ImgInfo(t,idx))
		else:
			raise KeyError('Could not find index for image '+t.name)
	# Sort
	img_infos.sort(key=lambda x: x.index, reverse=False)
	return img_infos

def get_img_from_tar(
	path: str,
	root : Optional[str] = '',
	index_file: Optional[str] = '',
	extensions: Optional[Tuple[str, ...]] = None,
) -> List[ImgInfo]:
	"""
	Open TAR file and returns a list of ImgInfo corresponding to the target images
	Args:
		path (string): Path to TAR archive
		root (string, optional): Root directory path for images inside archive
		index_file(string, optional): Path to file associating each image with an index
		extensions (tuple, optional): Tuple of allowed image extensions
	Returns:
		List of ImgInfo
	"""
	# Open TAR file with transparent compression
	tar = tarfile.open(path,mode='r')
	members = tar.getmembers()

	# Select files in root directory
	if root:
		members = [m for m in members
			if os.path.dirname(m.name).startswith(root)]
		# Add trailing '/' to root path (if necessary)
		if root[-1] != '/': root += '/'
		# Remove prefix from members' names
		for m in members:
			m.name = m.name[len(root):]

	# Check file extensions
	if extensions is not None :
		members = [m for m in members
			if m.name.lower().endswith(extensions)]

	# Use index file to sort images
	if index_file:
		# Open index file
		if index_file.startswith(path):
			# Index file is inside the TAR archive and given in the form
			# path/to/archive.tar/path/to/index_file
			index_fp = tar.extractfile(index_file[len(path)+1:])
			# Parse index file and sort images
			img_infos = sort_img(members,index_fp,decode=True)
		else:
			# Index is a simple file outside the TAR archive
			index_fp = open(index_file,'r')
			# Parse index file and sort images
			img_infos = sort_img(members,index_fp,decode=False)

	else :
		img_infos = [ImgInfo(m,i) for i,m in enumerate(members)]

	tar.close()
	return img_infos

def build_index_from_file(
	apath: str,
	img_infos: List[ImgInfo],
	lpath: str,
	preprocess: Optional[Callable] = None,
	ipath : Optional[str] = '',
) -> np.array:
	"""
	Find labels from file assuming the following content:
		image_index_0 list_of_label_information
		image_index_1 list_of_label_information
		...
	WARNING: To speed-up computation, assume that image indices are sorted
	Note: Images may have multiple labels described on several lines
	e.g.
		25 first_label_information
		25 second_label_information
		26 first_label_information
		26 second_label_information
	Then, for each image of the dataset, returns
		- its offset in the TAR archive
		- its size
		- its list of labels

	Args:
		path (string): Path to TAR archive
		img_infos (list): List of ImgInfo corresponding to the target images
		lpath(str): Path to file containing image labels
		preprocess (callable,optional): By default, the label is assumed to
			be a list of floats representing the image class index. However, it is
			possible to specify a preprocessing function taking the entire label
			information	string and returning a list of floats
		ipath (string, optional): Path to output file for index database
	Returns:
		Numpy array
	"""
	# Open the label file and read content
	if lpath.startswith(apath):
		# Label file is inside the TAR archive and given in the form
		# path/to/archive.tar/path/to/label_file
		tar = tarfile.open(apath,mode='r')
		label_fp = tar.extractfile(lpath[len(apath)+1:])
		lines = label_fp.read().splitlines()
		# Decode binary strings
		lines = [l.decode('utf-8') for l in lines]
	else:
		# Index is a simple file outside the TAR archive
		label_fp = open(lpath,'r')
		lines = label_fp.read().splitlines()

	# Init table
	data = []
	for r,img_info in enumerate(img_infos):
		np_data = [img_info.index,img_info.offset_data,img_info.size]
		data.append(np_data)

	# Current index in image table
	tab_idx = 0
	# Read all labels
	for l in lines:
		img_idx = int(l.split()[0])
		# Move on to the next image
		if img_idx > data[tab_idx][0]:	tab_idx += 1
		# Skip images not belonging to target set
		if img_idx < data[tab_idx][0]: continue
		if img_idx != data[tab_idx][0] :
			raise ValueError('Missing labels for image',str(data[tab_idx][0]))
		if preprocess is not None:
			data[tab_idx] += preprocess(l)
		else :
			data[tab_idx] += [np.float64(m) for m in l.split()[1:]]

	# Convert to numpy array
	data=np.array(data)

	if ipath:
		np.save(ipath,data,allow_pickle=True)
	return data

def build_index_from_directories(
	img_infos: List[ImgInfo],
	ipath : Optional[str] = '',
) -> np.array:
	"""
	Find classes assuming the following directory tree:
		class0/xxx.png
		class0/yyy.png
		class0/subdir/zzz.png
		...
		classN/ttt.png

	Note: all images images included in subdirectories of a class directory
	are considered part of the class

	Then, for each image of the dataset, returns
		- its offset in the TAR archive
		- its size
		- its class index

	Args:
		img_infos (list): List of ImgInfo corresponding to the target images
		ipath (string, optional): Path to output file for index database
	Returns:
		Numpy array
	"""
	# Find labels from directories
	classes = sorted(list(dict.fromkeys([
		os.path.dirname(t.name).split('/')[0] for t in img_infos])))
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

	data = []
	for i,t in enumerate(img_infos):
		cls_name = os.path.dirname(t.name).split('/')[0]
		cls_indx = class_to_idx[cls_name]
		data.append([t.index, t.offset_data, t.size, cls_indx])
	data = np.array(data)
	if ipath:
		np.save(ipath,data,allow_pickle=True)
	return data

def open_item (
	afile: BinaryIO,
	index: int,
	index_table: np.array) -> Tuple[BinaryIO,int]:
	"""
	Given an opened TAR archive, return item and class index
	Args:
		afile (BinaryIO): Opened TAR archive
		index (int): Index of item
		index_table (np.array): Table of item information
	Returns:
		Tuple containin file-like object pointing to item and class index
	"""
	data = index_table[index]
	offset = np.uint64(data[1])
	size   = np.uint64(data[2])
	labels = data[3:]
	# Move file pointer to offset
	afile.seek(offset)
	# Read item
	item = BytesIO(afile.read(size))
	item.seek(0)
	return item, labels

def pil_loader(buff: BinaryIO) -> Image.Image:
	img = Image.open(buff)
	return img.convert('RGB')

class ImageArchive:
	""" A data loader where the images are contained inside a TAR archive.
	For very large datasets, this class allows to manipulate a single file
	pointer per process rather than opening/closing the image files during
	training

	Args:
		apath (string): Path to TAR archive
		ipath (string, optional): Path to index file
		root (string, optional): Root path inside TAR archive
		image_index (string, optional): Path to file containing images indices
		image_label (string, optional): Path to file containing images labels
		image_label_preprocessing (callable, optional): A function that takes a
			string label and returns a float value
		transform (callable, optional): A function/transform that takes in an PIL
			image and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in
			the target and transforms it.
		loader (callable, optional): A function to load an image given its path.
		extensions (tuple of strings, optional): List of authorized file extensions
			inside the TAR archive
		data_in_memory (boolean, optional): Load content of TAR archive in memory
		open_after_fork (boolean, optional): When used in a multiprocess context,
			this option indicates that the TAR file should not be opened yet but
			rather after processes are spawned (using worker_open_archive method)
	"""

	def __init__(
			self,
			apath: str,
			ipath: Optional[str]= '',
			root:  Optional[str]= '',
			image_index: Optional[str]='',
			image_label: Optional[str]='',
			image_label_preprocessing: Optional[Callable] = None,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			loader: Callable[[BinaryIO], Any] = pil_loader,
			extensions: Optional[Tuple[str,...]] = None,
			data_in_memory: Optional[bool] = False,
			open_after_fork: Optional[bool]= False
	):
		# Using option image_label requires to specify image indices
		assert not(image_label) or image_index

		self.transform        = transform
		self.target_transform = target_transform
		self.loader           = loader

		############################

		if data_in_memory and open_after_fork :
			print('Ignoring open_after_fork option since archive is loaded in memory')
			open_after_fork = False

		self.data_in_memory  = data_in_memory
		self.open_after_fork = open_after_fork

		if data_in_memory:
			# Load entire archive into memory
			self.data = Path(apath).read_bytes()
		elif not(open_after_fork):
			# Open archive. This is safe only in a single process context
			self.afile = open(apath,'rb')
		else:
			# Store only path to file
			self.afile = None
			self.apath = apath

		############################

		if not(ipath):
			ipath = Path(os.path.splitext(apath)[0]+".idx.npy")

		if not(ipath.exists()):

			# Get list of TAR infos corresponding to all images of the dataset
			members = get_img_from_tar(apath,root,image_index,extensions)

			if image_label :
				self.idx = build_index_from_file(
					apath=apath,
					img_infos=members,
					lpath=image_label,
					preprocess=image_label_preprocessing,
					ipath=ipath)
			else:
				# Build index from directories
				self.idx = build_index_from_directories(members,ipath)

		else :
			self.idx = np.load(ipath,allow_pickle = True)
		self.nobjs = self.idx.shape[0]

	def __len__(self) -> int:
		"""
		Returns:
			Number of images in the database
		"""
		return self.nobjs

	def __getitem__ (self, index:int) -> Tuple[Any,Any]:
		"""
		Args:
			index (int): Index of the image in the database
		Returns:
			Tuple (image,class index)
		"""
		if self.data_in_memory:
			data = self.idx[index]
			offset = np.uint64(data[1])
			size   = np.uint64(data[2])
			label  = data[3:]
			item = BytesIO(self.data[offset:offset+size])
		else :
			item, label = open_item(self.afile,index,self.idx)
		item = self.loader(item)
		if self.target_transform is not None:
			label = self.target_transform(item,label)
		if self.transform is not None:
			item = self.transform(item)
		return item, label

	def worker_open_archive (self,wid):
		"""
		Explicitely open archive file (used in multiprocess context)
		"""
		if not(self.open_after_fork) or self.data_in_memory : return
		if not(self.afile):
			self.afile = open(self.apath,'rb')
