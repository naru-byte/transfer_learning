import os
from PIL import Image
import torch
from torchvision import transforms

def image_read(path):
    img = Image.open(path).convert("RGB")
    return img

def train_transform(x_size, y_size, mean, std):
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(y_size,x_size), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transformer

def valid_transform(x_size, y_size, mean, std):
    transformer = transforms.Compose([
        transforms.Resize(size=(y_size, x_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transformer

def make_loader(dataset, batch_size, shuffle=True):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader

class data_path:
    def __init__(self,data_dir, datasize):
        self.data_dir = data_dir
        self.datasize = datasize

    def __call__(self):
        return self.load_datas()

    def load_datas(self):
        t, v = self.datasize["train"], self.datasize["valid"]

        train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
        folders = sorted(os.listdir(self.data_dir))
        print(folders)
        for cnt, folder in enumerate(folders):
            images = sorted(os.listdir(os.path.join(self.data_dir,folder)))
            for image in images[:t]:
                train_x.append(os.path.join(*[self.data_dir,folder,image]))
                train_y.append(cnt)
            
            for image in images[t:t+v]:
                val_x.append(os.path.join(*[self.data_dir,folder,image]))
                val_y.append(cnt)

            for image in images[t+v:]:
                test_x.append(os.path.join(*[self.data_dir,folder,image]))
                test_y.append(cnt)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

class NStrainimageData(torch.utils.data.Dataset):

    def __init__(self, image_path_list, label_list, transform=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        #index image load
        img_path = self.image_path_list[index]
        img = image_read(path=img_path)

        img_transformed = self.transform(img)

        label = self.label_list[index]

        return img_transformed, label

class NSvalidationimageData(torch.utils.data.Dataset):

    def __init__(self, image_path_list, label_list, transform=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        #index image load
        img_path = self.image_path_list[index]
        img = image_read(path=img_path)

        img_transformed = self.transform(img)

        label = self.label_list[index]

        return img_transformed, label