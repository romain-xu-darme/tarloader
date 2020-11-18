# Documentation for the ImageArchive class
Class for loading Deep Learning databases contained in TAR archives.

This class aims at reducing the cost of opening/closing image files during
training by keeping a single file pointer to the TAR archive, or better, to
load the entire dataset in memory.
## Path to the TAR archive
The main argument for creating an ImageArchive object is the path to the target
TAR archive.
```
dataset = ImageArchive(apath=path/to/my/archive.tar)
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
By default, the object enumerate all files contained in the archive ending with
the following extensions:
 * .jpg
 * .jpeg
 * .png
 * .ppm
 * .bmp
 * .pgm
 * .tif
 * .tiff
 * .webp

It is however possible to filter the images by manually specifying a set of
authorized file extensions.
```
dataset = ImageArchive(
  apath=path/to/my/archive.tar,
  extensions=('.jpg','.jpeg')
)
```
## Handling folder tree
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
  apath=path/to/my/archive.tar,
  root='images/train/'
)
```

## Index table
The ImageArchive object uses an **index table** storing for each image:
 * its offset inside the TAR file
 * its size
 * its class index

By default, the object looks for the file _/path/to/my/archive.idx.npy_ and
creates it if necessary. It is however possible to specify the path to this
index file manually.
```
dataset = ImageArchive(
  apath=path/to/my/archive.tar,
  ipath=path/to/my/index
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
  apath=path/to/my/archive.tar,
  loader = my_loader
)
```
### Applying transformation on images and class index
It is possible to specify a set of transformations that should be applied to
the images and to the class index.
```
import torchvision.transforms as transforms
import torch

def index_to_one_hot(index):
  return torch.eye(200)[index]

transform = transforms.Compose([
 transforms.RandomResizedCrop(224),
 transforms.RandomHorizontalFlip(),
 transforms.ToTensor()])

dataset = ImageArchive(
  apath=path/to/my/archive.tar,
  transform=transform,
  target_transform=index_to_one_hot
)
```
**Note:** It is not possible for now to specify a *target\_transform* function
taking more arguments than the class index. In the example above, the total
number of classes should be known prior to defining the function.

## Loading dataset in memory
For small datasets, it is possible to load the entire TAR archive into memory,
saving a lot of time during training.
```
dataset = ImageArchive(
  apath=path/to/my/archive.tar,
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
  apath=path/to/my/archive.tar,
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

**Note:** Setting *data_in_memory=True* disabled the *open_after_fork* option
because the entire dataset is already loaded in memory and accessible
independently by all workers.




