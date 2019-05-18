import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from download_images import loadData
from preprocess_data import extractName, getDataLabels

import numpy as np

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomVerticalFlip(),  # randomly flip image vertically
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class HOUSEDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, data_dir2, transform, train):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir1: (string) directory containing the dataset
            data_dir2: (string) directory containing the second dataset (must have the same elements)
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.keep = ['TaxRateArea', 'totBuildingDataLines', 'EffectiveYearBuilt', 'SQFTmain',
                     'LandValue', 'Bedrooms', 'Bathrooms', 'zip_class']

        self.filenames = self.getFilenames(data_dir)
        self.filenames2 = self.getFilenames(data_dir2)

        self.data = getDataLabels()
        self.data = self.data[self.data["rowID"].isin(self.getIds())].reset_index(drop=True)

        self.labels = self.getImagesLabel(id_="rowID", y_id="log_total_value")
        self.info_vector = self.getInfoVector(id_="rowID", y_id="log_total_value")

        self.transform = transform
        self.train = train

        ## Uncomment to see the iamges at idx
#         idx = 11
#         image = Image.open(self.filenames[idx])  # PIL image
#         image2 = Image.open(self.filenames2[idx])  # PIL image
#         image.show()
#         image2.show()
#         vec  = self.info_vector.iloc[idx].to_list()
#         label = self.labels[idx]

#         print(vec, label)
#         display(self.data)

    def getFilenames(self, data_dir):
        filenames = os.listdir(data_dir)
        filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg')]
        return filenames

    def getIds(self):
        return list(map(extractName, self.filenames))

    def getImagesLabel(self, id_, y_id):
        ids = self.getIds()
        label = self.data[self.data[id_].isin(ids)][y_id].tolist()
        return label

    def getInfoVector(self, id_, y_id):
        ids = self.getIds()
        info_vector = self.data[self.data[id_].isin(ids)][self.keep].reset_index(drop=True) #.tolist()
        return info_vector


    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)

        image2 = Image.open(self.filenames2[idx])  # PIL image
        image2 = self.transform(image2)

        vec  = self.info_vector.iloc[idx].to_list()
        label = self.labels[idx]

        return image, image2, np.asarray(list(map(float,vec))), label


def fetch_dataloader(types, data_dir, data_dir2, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))
            path2 = os.path.join(data_dir2, "{}".format(split))


            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(HOUSEDataset(path, path2, train_transformer, train=True), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            if split == "val":
                dl = DataLoader(HOUSEDataset(path, path2, eval_transformer, train=False), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else:
                dl = DataLoader(HOUSEDataset(path, path2, eval_transformer, train=False), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders