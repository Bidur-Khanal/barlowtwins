import os
import os.path
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.datasets as datasets
import tifffile 
import h5py




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

def pil_loader_tifffile(path):
    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            img = page.asarray()
        img = Image.fromarray(img)
    return img.convert('RGB')




class custom_histopathology_classic(datasets.ImageFolder):

    """histopathology Dataset object: https://zenodo.org/record/1214456#.ZBf4GnbMKck
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """
    def __init__(self, root, transform, train = True):
        super().__init__(root, transform)
        self.root = root 
        self.transform = transform
        self.train = train
        self.loader = pil_loader_tifffile

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #print (path,target)
        sample = self.loader(path)
       
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
        
    def __len__(self) -> int:
        return len(self.samples)
    



class custom_COVID19_Xray_faster(data.Dataset):
    """COVID-QU-Ex Dataset object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root = "C:/Users/Station5/learning-with-noisy-labels-benchmark/data/", train=True, transform=None, target_transform=None, num_classes= 3, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"COVID-QU-Dataset/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

class custom_histopathology_faster(data.Dataset):
    """histopathology Dataset object: https://zenodo.org/record/1214456#.ZBf4GnbMKck
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "C:/Users/Station5/learning-with-noisy-labels-benchmark/data/", train=True, transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"histopathology/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        ### select only top 100 examples of each class, this is done for debugging only
        # all_targets = np.unique(self.targets)
        # curated_path_list = []
        # curated_target_list =[]
        # images = np.array(self.images)
        # targets = np.array(self.targets)
        # for i in all_targets:
        #     curated_path_list.extend(images[np.where(targets == i)][0:1000])
        #     curated_target_list.extend(targets[np.where(targets == i)][0:1000])
        # self.images = curated_path_list 
        # self.targets = curated_target_list

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    


class custom_FETAL_PLANE_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 6, seed=1):  

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"Fetal_plane/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class custom_dermnet_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 23, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"Dermnet/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    


class MURA_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        seed: random seed for shuffling classes or instances (default=10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root = "data/", train=True, transform=None, target_transform=None, num_classes= 7, seed=1):

        self.root = root
        self.as_rgb = True
        if train:
           self.mode = "train"
        else:
           self.mode = "valid"

        with h5py.File(os.path.join(root,"MURA/", str(self.mode)+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str





