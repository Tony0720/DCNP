import numpy as np
from PIL import Image
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    H,W=img.size
    img = img.resize((256,256), Image.BICUBIC)
    return img,H,W


def save_img(image_tensor, H,W,filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((H,W), Image.BICUBIC)
    image_pil.save(filename)
    #print("Image saved as {}".format(filename))


def save_feature_map(feature_map, filename):
    image_numpy = feature_map.float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((200, 250), Image.BICUBIC)
    image_pil.save(filename)


def tensor2img(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

def get_attention(fake_label, real_label):
    error_map = torch.abs(fake_label.detach() - real_label.detach()) * 0.5
    return error_map

def get_facial_label(facial_tensor):
    facial_label = torch.argmax(facial_tensor, 1).unsqueeze(1)
    facial_label = (facial_label.float() / 18.0 - 0.5) * 2.0
    return facial_label
