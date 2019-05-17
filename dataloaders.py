from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
from PIL import Image
import torch


class DenoisingDataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False):
        """
        Args:
            root_dirs (string): A list of directories with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('png','jpg','jpeg','bmp'))]
        self.verbose = verbose

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image


def get_dataloaders(train_path_list, test_path_list, crop_size=128, batch_size=1):
    batch_sizes = {'train': batch_size, 'test':1}

    train_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}
    image_datasets = {'train': DenoisingDataset(train_path_list, data_transforms['train']),
                      'test': DenoisingDataset(test_path_list, data_transforms['test'])}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=(x == 'train')) for x in ['train', 'test']}
    return dataloaders