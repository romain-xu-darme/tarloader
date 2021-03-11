import os
import tarfile
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path
from PIL import Image
from io import BytesIO

####################################################################################
# BEGIN: Extracted from torchvision.datasets.ImageFolder
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
	"""Checks if a file is an allowed extension.

	Args:
		filename (string): path to a file
		extensions (tuple of strings): extensions to consider (lowercase)

	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	return filename.lower().endswith(extensions)

def is_image_file(filename: str) -> bool:
	"""Checks if a file is an allowed image extension.

	Args:
		filename (string): path to a file

	Returns:
		bool: True if the filename ends with a known image extension
	"""
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)
# END: Extracted from torchvision.datasets.ImageFolder
####################################################################################

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
	is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[ImgInfo]:
	"""
	Open TAR file and returns a list of ImgInfo corresponding to the target images
	Args:
		path (string): Path to TAR archive
		root (string, optional): Root directory path for images inside archive
		index_file(string, optional): Path to file associating each image with an index
		extensions (tuple[string]): A list of allowed extensions.
			both extensions and is_valid_file should not be passed.
		is_valid_file (callable, optional): A function that takes path of a file
			and check if the file is a valid file (used to check of corrupt files)
			both extensions and is_valid_file should not be passed.
	Returns:
		List of ImgInfo
	"""
####################################################################################
# BEGIN: Extracted from torchvision.datasets.ImageFolder
	both_none = extensions is None and is_valid_file is None
	both_something = extensions is not None and is_valid_file is not None
	if both_none or both_something:
		raise ValueError(
			"Both extensions and is_valid_file cannot be None or not None at the same time")
	if extensions is not None:
		def is_valid_file(x: str) -> bool:
			return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
	is_valid_file = cast(Callable[[str], bool], is_valid_file)
# END: Extracted from torchvision.datasets.ImageFolder
####################################################################################

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
	members = [m for m in members if is_valid_file(m.name)]

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
			information string and returning a list of floats
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
####################################################################################
# BEGIN: Extracted from torchvision.datasets.ImageFolder
	class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
# END: Extracted from torchvision.datasets.ImageFolder
####################################################################################

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
	Given an opened TAR archive, return item and label
	Args:
		afile (BinaryIO): Opened TAR archive
		index (int): Index of item
		index_table (np.array): Table of item information
	Returns:
		Tuple containin file-like object pointing to item and label
	"""
	data = index_table[index]
	abs_idx= int(data[0])
	offset = np.uint64(data[1])
	size   = np.uint64(data[2])
	labels = data[3:]
	# Move file pointer to offset
	afile.seek(offset)
	# Read item
	item = BytesIO(afile.read(size))
	item.seek(0)
	return item, abs_idx, labels

####################################################################################
# BEGIN: Extracted from torchvision.datasets.ImageFolder
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(buff: BinaryIO) -> Image.Image:
	img = Image.open(buff)
	return img.convert('RGB')
# END: Extracted from torchvision.datasets.ImageFolder
####################################################################################

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
		split_mask (np array, optional): Splitting mask associating a value to
			each image
		split_val (int, optional): Splitting value.
		transform (callable, optional): A function/transform that takes in an PIL
			image and returns a transformed version. E.g, ``transforms.RandomCrop``
		index_transform (callable, optional): A function/transform that takes in an PIL
			image and the image index and returns a transformed version (used to apply index
			based transforms)
		target_transform (callable, optional): A function/transform that takes in
			the target and transforms it.
		batch_size (int, optional): Size of each batch of data. If set to 1 (default), the
			generator returns individual items of the dataset
		drop_last (bool, optional): Drop last non complete batch (if any)
		loader (callable, optional): A function to load an image given its path.
		is_valid_file (callable, optional): A function that takes path of an Image file
			and check if the file is a valid file (used to check of corrupt files)
		data_in_memory (boolean, optional): Load content of TAR archive in memory
		open_after_fork (boolean, optional): When used in a multiprocess context,
			this option indicates that the TAR file should not be opened yet but
			rather after processes are spawned (using worker_open_archive method)
		overwrite_index (boolean, optional): Overwrite index file if present
	"""

	def __init__(
			self,
			apath: str,
			ipath: Optional[str]= '',
			root:  Optional[str]= '',
			image_index: Optional[str]='',
			image_label: Optional[str]='',
			image_label_preprocessing: Optional[Callable] = None,
			split_mask: Optional[np.array] = None,
			split_val: Optional[int] = 0,
			transform: Optional[Callable] = None,
			index_transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			batch_size: Optional[int] = 1,
			drop_last: Optional[bool] = False,
			loader: Callable[[BinaryIO], Any] = pil_loader,
			is_valid_file: Optional[Callable[[str], bool]] = None,
			data_in_memory: Optional[bool] = False,
			open_after_fork: Optional[bool]= False,
			overwrite_index: Optional[bool]= False,
	):
		# Using option image_label requires to specify image indices
		if image_label and not(image_index):
			raise ValueError(
				'Specifying image_label file requires to also specify image_index file')

		self.transform        = transform
		self.index_transform  = index_transform
		self.target_transform = target_transform
		self.batch_size       = batch_size
		self.drop_last        = drop_last
		self.loader           = loader
		############################

		if data_in_memory and open_after_fork :
			print(f'[{self.__class__.__name__}] Ignoring open_after_fork option (archive loaded in memory)')
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
			ipath = os.path.splitext(apath)[0]+".idx.npy"
		ipath = Path(ipath)

		if not(ipath.exists()) or overwrite_index:
			print(f'[{self.__class__.__name__}] Building index file {ipath}')
			extensions = IMG_EXTENSIONS if is_valid_file is None else None

			# Get list of TAR infos corresponding to all images of the dataset
			members = get_img_from_tar(apath,root,image_index,extensions,is_valid_file)

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
			print(f'[{self.__class__.__name__}] Loading index file {ipath}')
			self.idx = np.load(ipath,allow_pickle = True)

		# Split dataset
		if split_mask is not None:
			self.idx = self.idx[split_mask==split_val]

		self.nobjs = self.idx.shape[0]

	def __len__(self) -> int:
		"""
		Returns:
			Number of images/batches in the database
		"""
		nbatches = int(self.nobjs/self.batch_size)
		if (self.nobjs % self.batch_size > 0) and not(self.drop_last):
			nbatches += 1
		return nbatches

	def __getsingleitem (self, index:int) -> Tuple[Any,Any]:
		"""
		Args:
			index (int): Index of the image in the database
		Returns:
			Tuple (image,label)
		"""
		if self.data_in_memory:
			data = self.idx[index]
			abs_idx= int(data[0])
			offset = np.uint64(data[1])
			size   = np.uint64(data[2])
			label  = data[3:]
			item = BytesIO(self.data[offset:offset+size])
		else :
			if self.afile is None:
				raise LookupError(f'[{self.__class__.__name__}] Archive file not opened yet.'\
					'This may happen when you use open_after_fork option but forgot to call '\
					'worker_open_archive in each worker in a multiprocess context.')
			item, abs_idx, label = open_item(self.afile,index,self.idx)
		item = self.loader(item)
		if self.target_transform is not None:
			label = self.target_transform(item,label)
		if self.index_transform is not None:
			item = self.index_transform(item,abs_idx)
		if self.transform is not None:
			item = self.transform(item)
		return item, label

	def __getitem__ (self, index:int) -> Tuple[Any,Any]:
		"""
		Args:
			index (int): Index of the image/batch in the database
		Returns:
			Tuple (image,label) or Tuple(array(images),array(labels)) depending on batch size
		"""
		if index < 0: index = index % self.__len__()
		if index >= self.__len__(): raise StopIteration
		if self.batch_size == 1:
			return self.__getsingleitem(index)
		else:
			batch_start = index*self.batch_size
			batch_end   = min((index+1)*self.batch_size,self.nobjs)
			X = []
			Y = []
			for idx in range(batch_start,batch_end):
				x,y = self.__getsingleitem(idx)
				X.append(x)
				Y.append(y)
			return np.array(X),np.array(Y)

	def worker_open_archive (self,wid):
		"""
		Explicitely open archive file (used in multiprocess context)
		"""
		if not(self.open_after_fork) or self.data_in_memory : return
		if not(self.afile):
			self.afile = open(self.apath,'rb')

def ImageArchive_add_parser_options(parser):
	""" Add all common arguments for the (basic) initialization of a
	ImageArchive object
	Args:
		parser: argparse argument parser
	"""
	parser.add_argument('--tarloader-archive', type=str,required=True,
		metavar=('<path_to_file>'),
		help='Path to the TAR archive.')
	parser.add_argument('--tarloader-index-file', type=str,required=False,
		metavar=('<path_to_file>'),
		help='Path to the Numpy file containing index.')
	parser.add_argument('--tarloader-archive-root-directory', type=str,required=False,
		default="images",
		metavar=('<path_to_directory>'),
		help='Path to root directory inside TAR archive.')
	parser.add_argument('--tarloader-image-file', type=str, required=False,
		metavar=('<path_to_file>'),
		help='Path to list of images.')
	parser.add_argument('--tarloader-index-file-overwrite', required=False,
		action='store_true',
		help='Overwrite index (if it exists)')
	parser.add_argument('--tarloader-label-file', required=False,
		metavar=('<path_to_file>'),
		help='Path to label file.')
	parser.add_argument('--tarloader-batch-size', type=int, required=False,
		metavar=('<size>'),
		default=1,
		help='Batch size')
	parser.add_argument('--tarloader-drop-last-batch', required=False,
		action='store_true',
		help='Drop last non complete batch (if any)')
	parser.add_argument('--tarloader-keep-in-memory', required=False,
		action='store_true',
		help='Keep all dataset in memory')
	parser.add_argument('--tarloader-enable-multiprocess', required=False,
		action='store_true',
		help='Enable multiprocess mode')
