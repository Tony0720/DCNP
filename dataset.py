from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.random_filenames = self.image_filenames[:]
        random.shuffle(self.random_filenames)
        # self.b_image_filenames = [x for x in listdir(self.b_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    # def __getitem__(self, index):
    #     # Load Image
    #     a = load_img(join(self.a_path, self.image_filenames[index]))
    #     a = self.transform(a)
    #     b = load_img(join(self.b_path, self.image_filenames[index]))
    #     b = self.transform(b)

    #     return a, b

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        # index_b = random.randint(0, len(self.b_image_filenames) - 1)
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((286,286), Image.BICUBIC)
        b = b.resize((286,286), Image.BICUBIC)

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        a0 = Image.open(join(self.a_path, self.random_filenames[index])).convert('RGB')
        # index_b = random.randint(0, len(self.b_image_filenames) - 1)
        b0 = Image.open(join(self.b_path, self.random_filenames[index])).convert('RGB')
        a0 = a0.resize((286,286), Image.BICUBIC)
        b0 = b0.resize((286,286), Image.BICUBIC)

        a0 = transforms.ToTensor()(a0)
        b0 = transforms.ToTensor()(b0)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a0 = a0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b0 = b0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        a0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a0)
        b0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b0)

        return a, b, a0, b0

    def __len__(self):
        return len(self.image_filenames)
