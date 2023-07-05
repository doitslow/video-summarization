import os
import collections
import numpy as np
from os.path import join
from shutil import copy2


def filter_doubles(scores, keywords, img_dir, img_files, out_dir, topk=5, caps=None, save_img=False):
    topk_inds = np.zeros((scores.shape[0], topk), dtype=int)
    topk_scores = np.zeros((scores.shape[0], topk))
    for i in range(scores.shape[0]): # for each keyword
        ith_scores = scores[i, :].squeeze()
        inds = np.argpartition(ith_scores, -topk)[-topk:]
        topk_inds[i, :] = inds
        topk_scores[i, :] = ith_scores[inds]
    counts = collections.Counter(list(topk_inds.flatten()))
    top_dups = [k for k, v in sorted(counts.items(), key=lambda item: item[1])][-10:]
    # dups = [item for item, count in collections.Counter(list(topk_inds.flatten())).items() if count > 1]

    reasons = []
    for dup in top_dups:
        locs = np.where(topk_inds == dup)
        c_keywords = [keywords[i] for i in list(locs[0])]
        selected_scores = topk_scores[locs]
        img_file = img_files[topk_inds[locs][0]]
        if save_img:
            copy2(join(img_dir, img_file), out_dir)
        selected_scores = ["{:.4f}".format(score) for score in selected_scores]
        caption = caps[topk_inds[locs][0]] if caps else 'No caption'
        reason = '\t'.join([img_file, caption, '; '.join(c_keywords), '; '.join(selected_scores)])
        print(reason)
        reasons.append(reason)

    return reasons


def write_reason(reasons, out_file, prefix):
    with open(out_file, 'w') as fopen:
        if isinstance(prefix, list):
            fopen.write("Keywords of description are: " + "\n")
            fopen.write(";\t".join(prefix) + '\n')
            fopen.write('\n')
        else:
            fopen.write("Keywords' sentence of description is: " + "\n")
            fopen.write(prefix + '\n')
            fopen.write('\n')
        for item in reasons:
            fopen.write(item + '\n')
        fopen.close()


