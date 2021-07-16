from typing import Any, Callable, cast, Dict, List, Optional, Tuple, BinaryIO
from tarloader import ImageArchive,ImageArchive_add_parser_options,pil_loader
from tensorflow.keras.utils import Sequence
import numpy as np

class KerasImageArchive (Sequence):
	""" Keras wrapper for the ImageArchive dataloader
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
		self.dataset = ImageArchive(
			apath = apath,
			ipath = ipath,
			root  = root,
			image_index = image_index,
			image_label = image_label,
			image_label_preprocessing = image_label_preprocessing,
			split_mask = split_mask,
			split_val = split_val,
			transform = transform,
			index_transform = index_transform,
			target_transform = target_transform,
			batch_size = batch_size,
			drop_last = drop_last,
			loader = loader,
			is_valid_file = is_valid_file,
			data_in_memory = data_in_memory,
			open_after_fork = open_after_fork,
			overwrite_index = overwrite_index
		)

	def __len__(self) -> int:
		"""
		Returns:
			Number of images/batches in the database
		"""
		return len(self.dataset)

	def __getitem__ (self, index:int) -> Tuple[Any,Any]:
		"""
		Args:
			index (int): Index of the image/batch in the database
		Returns:
			Tuple (image,label) or Tuple(array(images),array(labels)) depending on batch size
		"""
		return self.dataset[index]

	def worker_open_archive (self):
		"""
		Explicitely open archive file (used in multiprocess context)
		"""
		self.dataset.worker_open_archive(0)

def KerasImageArchive_add_parser_options(parser):
	""" Add all common arguments for the (basic) initialization of a
	KerasImageArchive object
	Args:
		parser: argparse argument parser
	"""
	ImageArchive_add_parser_options(parser)
