import os
import argparse
from os.path import join, isfile, isdir, basename, dirname, exists
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from shutil import copy2
import numpy as np

from .util import filter_doubles, write_reason
from ..utils import load_txt, calc_sim, read_caps
from sentence_transformers import SentenceTransformer, util

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class VideoFrames(Dataset):
    def __init__(self, root, img_list, transform):
        self.transform = transform
        # self.paths = sorted([join(root, p) for p in os.listdir(root) if isfile(join(root, p))])
        # self.paths = sorted([join(root, line) for line in open(list_file, "r").read().splitlines() if line][1:])
        self.paths = [join(root, f) for f in img_list]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = Image.open(self.paths[item])
        # return self.transform(image).unsqueeze(0)
        image = self.transform(image)
        name = basename(self.paths[item])
        # save_image(image.squeeze(), join('/home/john/Desktop/test', name))

        return image, name


def extract_img_feats(text, img_dir, img_list, batch_size):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    # Prepare the inputs
    dataset = VideoFrames(img_dir, img_list, _transform([224, 224]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    text_input = clip.tokenize(text).to(device)
    # Calculate features
    all_img_feats = torch.zeros(len(dataset), 512, dtype=torch.float16).to(device)
    all_fnames = []
    max_iter = len(dataset) // batch_size
    with torch.no_grad():
        text_feature = model.encode_text(text_input)
        for batch, (imgs, fnames) in enumerate(data_loader):
            feats = model.encode_image(imgs.to(device))
            if batch <= max_iter:
                all_img_feats[batch * batch_size:(batch + 1) * batch_size, :] = feats
            else:
                all_img_feats[batch * batch_size:, :] = feats
            all_fnames.extend(fnames)

    all_img_feats /= all_img_feats.norm(dim=-1, keepdim=True)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    cos_scores = torch.mm(text_feature, all_img_feats.transpose(0, 1))
    cos_scores = cos_scores.cpu().detach().numpy().squeeze()
    cos_scores = (cos_scores + 1) / 2
    print("The max/min similarity scores is: {:.4f}/{:.4f}"
          .format(np.max(cos_scores), np.min(cos_scores)))

    torch.cuda.empty_cache()

    return cos_scores, all_fnames


def keysen_to_imgs(keysen, img_dir, img_list, out_dir, batch_size, topk=10, save_img=False):
    keysentence = [keysen.strip()]
    cos_scores, fnames = extract_img_feats(
        keysentence, img_dir, img_list, batch_size
    )
    topk_inds = np.argpartition(cos_scores, -topk)[-topk:]
    reasons = []
    for i in topk_inds.tolist():
        reason = '\t'.join([fnames[i], 'No caption', keysentence[0], "{:.4f}".format(cos_scores[i])])
        if save_img:
            copy2(join(img_dir, fnames[i]), out_dir)
        reasons.append(reason)
    write_reason(reasons, join(out_dir, 'reasoning.txt'), keysentence)

    return cos_scores, reasons


def keys_to_imgs(keys, img_dir, img_list, out_dir, batch_size, topk=10, save_img=False):
    keys = [key.strip() for key in keys]
    cos_scores, fnames = extract_img_feats(keys, img_dir, img_list, batch_size)
    assert img_list == fnames, "Loading data changed order of images!"
    reasons = filter_doubles(
        cos_scores, keys, img_dir, fnames, out_dir, topk, save_img=save_img
    )
    write_reason(reasons, join(out_dir, 'reasoning.txt'), keys)

    return cos_scores, reasons


def get_args():
    parser = argparse.ArgumentParser('Extract images selected by hecate')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--img_dir', '-i', type=str,
                        help='path to all the images')
    args = parser.parse_args()
    return args


