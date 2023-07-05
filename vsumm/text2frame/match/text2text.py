import os
import torch
import collections
import numpy as np
from shutil import copy2
from os.path import join, basename, dirname, exists

from sentence_transformers import SentenceTransformer, util
from ..utils import load_txt, calc_sim, read_caps
from .util import write_reason, filter_doubles


def extract_feats(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings1, embeddings2)
    cos_scores = cos_scores.cpu().detach().numpy().squeeze()
    cos_scores = (cos_scores + 1) / 2
    print("The max/min similarity scores comparing captions to descriptions"
          " is: {:.4f}/{:.4f}".format(np.max(cos_scores), np.min(cos_scores)))

    torch.cuda.empty_cache()

    return cos_scores


def keysen_to_caps(keysen, image_dir, caption_file, out_dir, topk=10, save_img=False):
    caps, caps_ind = read_caps(caption_file)
    keysen = [key.strip() for key in [keysen]]
    cos_scores = extract_feats(keysen, caps)
    topk_inds = np.argpartition(cos_scores, -topk)[-topk:]

    reasons = []
    for ind in topk_inds:
        img = caps_ind[ind]
        if save_img:
            copy2(join(image_dir, img), out_dir)
        # reason = "{}\t Caption:{}\t Score: {:.4f}".format(img, caps[ind], cos_scores[ind])
        reason = '\t'.join([img, caps[ind], keysen[0], "{:.4f}".format(cos_scores[ind])])
        reasons.append(reason)
    write_reason(reasons, join(out_dir, 'reasoning.txt'), keysen)

    return cos_scores, reasons


def keys_to_caps(keys, image_dir, caption_file, out_dir, topk=5, save_img=False):
    caps, caps_ind = read_caps(caption_file)
    keys = [key.strip() for key in keys]
    cos_scores = extract_feats(keys, caps)
    reasons = filter_doubles(
        cos_scores, keys, image_dir, caps_ind, out_dir, topk, caps, save_img)
    write_reason(reasons, join(out_dir, 'reasoning.txt'), keys)

    return cos_scores, reasons
