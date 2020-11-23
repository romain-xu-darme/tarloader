# Documentation for the ImageArchive class
Class for loading Deep Learning databases contained in TAR archives.

This class aims at reducing the cost of opening/closing image files during
training by keeping a single file pointer to the TAR archive, or better, to
load the entire dataset in memory.
## Path to the TAR archive
The main argument for creating an ImageArchive object is the path to the target
TAR archive.
```
dataset = ImageArchive(apath='path/to/my/archive.tar')
```
In this setup, the object assumes that the content of the TAR archive is as
follows:
```
class_name_0/xxx.png
class_name_0/yyy.png
...
class_name_N/zzz.png
```
All classes names are sorted alphabetically and associated with an index
ranging from 0 to N.

## Handling file extensions
By default, the object enumerate all files contained in the archive.
It is however possible to filter the images by manually specifying a set of
authorized file extensions.
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  extensions=('.jpg','.jpeg')
)
```
## Handling the folder tree
Assuming that the TAR archive does contain several subdirectories, it is
possible to specify the internal path to the target images.

ex. Assuming that the content of the TAR archive is organized as follows
```
README
docs/
images/
 |- train/
 |- test/
 |- val/
```
it is possible to load only images from the images/train directory.
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  root='images/train/'
)
```

## Specifying image indices and labels
Instead of infering labels from the folder tree, it is possible to explicitely
define:
 * the index associated with each image
 * the label(s) associated with each image
### Image index file format
The format of the index file should be as follows:
<image_index_value> <path/to/image/inside/root/directory>
```
/data/CUB-200/$ head -n 5 images.txt
1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
2 001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg
3 001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg
4 001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.jpg
5 001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg
```
It is possible to specify the path to the index file **inside or outside** the
TAR archive.
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  root='images/',
  # Path inside archive
  image_index='path/to/my/archive.tar/path/to/file/inside/archive'
)
```
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  root='images/',
  # Path outside archive
  image_index='path/to/my/index/file'
)
```
### Image labels file format
The format of the label file should be as follows:
<image_index_value> <list_of_labels>
```
/data/CUB-200$ head -n 5 bounding_boxes.txt
1 60.0 27.0 325.0 304.0
2 139.0 30.0 153.0 264.0
3 14.0 112.0 388.0 186.0
4 112.0 90.0 255.0 242.0
5 70.0 50.0 134.0 303.0
```
It is also possible to have multiple lines of labels for the same image index.
```
/data/CUB-200$ head -n 5 attributes/image_attribute_labels.txt
1 1 0 3 27.7080
1 2 0 3 27.7080
1 3 0 3 27.7080
1 4 0 3 27.7080
1 5 1 3 27.7080
```
By default, the object concatenate all labels for a given image index inside a
list of float. It is however possible to specify a preprocessing function that
will be applied on each line of the label file.

```
def preprocess_attributes (raw_data):
  # Given a line in the form
  # <image_id> <attribute_id> <is_present> <certainty_id> <time>
  # only return -1 or 1 depending on presence
  is_present = raw_data.split()[2]
  res = -1 if is_present == 0 else 1
  return [res]

dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  root='images/',
  # Path outside archive
  image_index='data/CUB-200.tar/images.txt',
  image_label='data/CUB-200/attributes/image_attribute_labels.txt',
  image_label_preprocessing=preprocessing_attributes
)

```
Note: The preprocessing function should always return a list of float.

## Index table
The ImageArchive object uses an **index table** storing for each image:
 * its index
 * its offset inside the TAR file
 * its size
 * its list of labels stored as a list of floats

By default, the object looks for the file _/path/to/my/archive.idx.npy_ and
creates it if necessary. It is however possible to specify the path to this
index file manually.
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  ipath='path/to/my/index'
)

```

## Image loader and transformations
### Loading images
By default, each image is loaded using the _open_ function of the PIL.Image
module. It is possible to specify a custom loader function but, **contrary**
**to common loaders, the specified function should take a file pointer as**
**input**.
```
def my_loader(file: BinaryIO):
  img = my_module.open(file)
  ...
  return img

dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  loader = my_loader
)
```
### Applying transformation on images and class index
It is possible to specify a set of transformations that should be applied to
the images and to the class index.
```
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

def rescale_bbox(
  img:Image.Image,
  labels:List
):
  """
  Rescale bounding box to range [0,1] with respect to image
  original dimensions
  """
  src_width  = img.size[0]
  src_height = img.size[1]
  return np.array([
    labels[0]/src_width,
    labels[1]/src_height,
    labels[2]/src_width,
    labels[3]/src_height])

transform = transforms.Compose([
 transforms.RandomResizedCrop(224),
 transforms.RandomHorizontalFlip(),
 transforms.ToTensor()])

dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  transform=transform,
  target_transform=index_to_one_hot
)
```
**Note:** The *target\_transform* function should always take an image and
a list of labels as inputs and it will be applied using the input image
**before** *transform* is applied. This allows, for example, to rescale a
bounding box before the image is resized.

## Loading dataset in memory
For small datasets, it is possible to load the entire TAR archive into memory,
saving a lot of time during training.
```
dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  data_in_memory=True
)
```
Regardless of this mode, the object will **always keep the index table in**
**memory**.

## Keeping the file opened and multiprocess
For larger datasets, the object will still optimize the number of file
manipulations by keeping an opened file pointer to the TAR archive file rather
than opening/closing the archive to retrieve each image.

However, in a multiprocess context during training, several workers may attempt
to manipulate the same file pointer to retrieve their images, causing data
corruption. In order to avoid this scenario, it is possible to explicitely ask
the object **not to open the TAR archive during its construction**, but rather
to wait after the processes are forked so that each process will handle its
personal file pointer. In this setup, it is necessary to use callback functions
or hooks to ensure that each process calls the function *worker\_open\_archive*
and to enable the *open_after_fork* mode.
```
import torch.utils.data
import torchvision.transforms as transforms

transform = transforms.Compose([
  transforms.RandomResizedCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor()
])

train_dataset = ImageArchive(
  apath='path/to/my/archive.tar',
  transform = transform,
  data_in_memory = False,
  open_after_fork = True)

worker_init_fn = train_dataset.worker_open_archive if open_after_fork else None

train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=64, shuffle=False,
  num_workers=10, worker_init_fn=train_dataset.worker_open_archive,
  pin_memory=True
)
```
**Note:** Using Tensorflow, it should be possible to use the *callbacks* option
of the *fit* function to call the *worker_open_archive* for each worker.

**Note:** Setting *data_in_memory=True* disables the *open_after_fork* option
because the entire dataset is already loaded in memory and accessible
independently by all workers.




