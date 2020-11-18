import os
import tarfile
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path
from PIL import Image
from io import BytesIO

IMG_EXTENSIONS = (
	'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_img_from_tar(
	path: str,
	root : Optional[str] = '',
	extensions: Optional[Tuple[str, ...]] = None,
) -> List[tarfile.TarInfo]:
	"""
	Open TAR file and returns a list of TarInfo corresponding to the target images
	Args:
		path (string): Path to TAR archive
		root (string, optional): Root directory path inside archive
		extensions (tuple, optional): Tuple of allowed image extensions
	Returns:
		List of TarInfo
	"""
	# Open TAR file with transparent compression
	tar = tarfile.open(path,mode='r')
	members = tar.getmembers()
	tar.close()
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
	return members

def find_classes(
	tar_infos: List[tarfile.TarInfo],
) -> Dict[str,int]:
	"""
	Find classes assuming the following directory tree:
		class0/xxx.png
		class0/yyy.png
		class0/subdir/zzz.png
		...
		classN/ttt.png

	Note: all images images included in subdirectories of a class directory
	are considered part of the class

	Args:
		tar_infos (list): List of TarInfos corresponding to the target images
	Returns:
		Dictionnary of class names and their corresponding index
	"""
	classes = sorted(list(dict.fromkeys([
		os.path.dirname(t.name).split('/')[0] for t in tar_infos])))
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
	return class_to_idx

def build_index(
	tar_infos: List[tarfile.TarInfo],
	class_to_idx: Dict[str,int],
	ipath : Optional[str] = '',
) -> np.array:
	"""
	For each image of the dataset, returns
		- its offset in the TAR archive
		- its size
		- its class index

	Args:
		tar_infos (list): List of TarInfos corresponding to the target images
		class_to_idx (dict): Dictionary associating each class to an index
		ipath (string, optional): Path to output file for index database
	Returns:
		Numpy array
	"""
	idx = np.empty((len(tar_infos),3),dtype=np.uint64)
	for i,t in enumerate(tar_infos):
		cls_name = os.path.dirname(t.name).split('/')[0]
		cls_indx = class_to_idx[cls_name]
		idx[i] = t.offset_data, t.size, cls_indx

	if ipath:
		np.save(ipath,idx,allow_pickle=True)
	return idx

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
	offset, size, cls_indx = index_table[index]
	# Move file pointer to offset
	afile.seek(offset)
	# Read item
	item = BytesIO(afile.read(size))
	item.seek(0)
	# Cast class index as int64 because some backends don't like uint64
	return item, np.int64(cls_indx)

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
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			loader: Callable[[str], Any] = pil_loader,
			extensions: Optional[Tuple[str,...]] = None,
			data_in_memory: Optional[bool] = False,
			open_after_fork: Optional[bool]= False
	):
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
			self.apath = apath

		############################

		if not(ipath):
			ipath = Path(os.path.splitext(apath)[0]+".idx.npy")

		if not(ipath.exists()):

			if extensions is None :
				extensions = IMG_EXTENSIONS

			# Get list of TAR infos corresponding to all images of the dataset
			members = get_img_from_tar(apath,root,extensions)
			# Find class names and index
			class_to_idx = find_classes(members)
			# Build index
			self.idx = build_index(members,class_to_idx,ipath)

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
			offset, size, cls_indx = self.idx[index]
			item = BytesIO(self.data[offset:offset+size])
		else :
			item, cls_indx = open_item(self.afile,index,self.idx)
		item = self.loader(item)
		if self.transform is not None:
			item = self.transform(item)
		if self.target_transform is not None:
			cls_indx = self.target_transform(cls_indx)
		return item, cls_indx

	def worker_open_archive (self,wid):
		"""
		Explicitely open archive file (used in multiprocess context)
		"""
		if not(self.open_after_fork) or self.data_in_memory : return
		self.afile = open(self.apath,'rb')
