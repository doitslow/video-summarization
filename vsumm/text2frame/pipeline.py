import os
import sys
# print(sys.path)
import argparse
from os.path import join, basename, exists, dirname
import numpy as np
from shutil import copy2

# from utils import create_description_file, count_jpeg, read_description_file, \
#     check_list_file, write_list_file
from .keywords.yake_keywords import yake1file
from .matching.text2image import keysen_to_imgs, keys_to_imgs
from .matching.text2text import keysen_to_caps, keys_to_caps
from .matching.util import filter_doubles, write_reason
from .cleaning.similar_images import remove_similar
from .cleaning.open_end import OpenEnd
from .caption.caption_batch import caption_frames
from .tools.hecate import get_hecate_images
from .tools.vid2img import vid_to_frame
from .utils import create_description_file, count_jpeg, read_description_file, \
    check_list_file, write_list_file, create_dir


class SemanticMatch(object):
    def __init__(self, args):
        self.args = args
        self._work_dir = None
        self._temp_dir = None
        self._out_dir = None

    @property
    def work_dir(self):
        return self._work_dir

    @work_dir.setter
    def work_dir(self, din):
        self._work_dir = din

    @property
    def temp_dir(self):
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, din):
        create_dir(din)
        self._temp_dir = din

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, din):
        create_dir(din)
        self._out_dir = din

    def get_keywords(self, video_path): # STEP1: extract keywords
        if not exists(join(self.work_dir, 'description.txt')):
            print(self.args.meta_path)
            if not create_description_file(video_path, self.args.meta_path):
                print("Semantic matching cannot be done as description file cannot be found nor created !!!")
                return
        description, keywords, keywords_in_sentence = read_description_file(
            join(self.work_dir, 'description.txt'))
        if keywords is None or keywords_in_sentence is None:
            keywords, keysen = yake1file(join(self.work_dir, 'description.txt'))
        else:
            keywords = keywords.split('\t')[-1].split(';')
            keysen = keywords_in_sentence.split('\t')[-1]
        print("STEP1: Keywords generation done.")

        return keywords, keysen

    def vid_to_frame(self, video_path):
        if self.args.post_hecate:
            img_dir = join(self.work_dir, "post_hecate")
            if not get_hecate_images(video_path, self.args.aes_path, join(self.work_dir, 'post_hecate')):
                print("Semantic matching cannot be done as Hecate images cannot be fetched !!!")
                return
        else:
            img_dir = join(self.work_dir, "images_every_{}s".format(self.args.gap_in_seconds))
            vid_to_frame(video_path, img_dir, self.args.gap_in_seconds)

        return img_dir

    def cleaning(self, video_path, img_dir):
        # Clean away unwanted images and create a list of remaining
        if self.args.cleaning is None:
            list_file = join(self.temp_dir, basename(img_dir) + '.txt')
            if not check_list_file(list_file):
                write_list_file(list_file, sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')]))
        else:
            list_file = join(self.temp_dir, basename(img_dir) + '-post_{}_cleaning.txt'.format(self.args.cleaning))
            inds_to_remove = []
            if exists(list_file) and check_list_file(list_file):
                pass
            else:
                # remove similar images
                inds_to_remove.extend(remove_similar(img_dir, batch_size=self.args.clean_batch))  # returns a list
                # remove opening/ending
                if self.args.cleaning == 'c2':
                    cleaner = OpenEnd(video_path, self.temp_dir, batch_size=self.args.clean_batch)
                    open_removals, end_removals = cleaner.do_remove()
                    all_hecate_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])
                    for ind_, img in enumerate(all_hecate_images):
                        frame_id = int(img.replace('.jpeg', ''))
                        if open_removals[0] and open_removals[1]:
                            if open_removals[0] <= frame_id <= open_removals[1]:
                                print("Removing opening frame: ", frame_id)
                                inds_to_remove.append(ind_)
                        if end_removals[0] and end_removals[1]:
                            if end_removals[0] <= frame_id <= end_removals[1]:
                                print("Removing ending frame: ", frame_id)
                                inds_to_remove.append(ind_)

                uniq_inds_to_remove = list(set(inds_to_remove))
                all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])
                if len(uniq_inds_to_remove) == 0:
                    print("NOT removing any images in preprocess")
                    images_post_clean = all_images
                else:
                    images_post_clean = list(
                        np.delete(np.asarray(all_images), np.asarray(uniq_inds_to_remove), None))
                write_list_file(list_file, images_post_clean)
        print("STEP2: Finalising candidate frames done.")

        return list_file

    def run_single_video(self, video):
        self.work_dir = dirname(video)
        self.temp_dir = join(dirname(video), 'by-product')
        print("Working on {} with {}-{}".format(
            self.work_dir, 'hecate' if self.args.post_hecate else 'gap', self.args.method))
        # video_files = [d for d in os.listdir(self.work_dir) if d.endswith('.mp4')]
        # assert len(video_files) > 0, "No video files found in {}".format(self.work_dir)
        # assert len(video_files) == 1, "More than 1 video files found in {}".format(self.work_dir)
        # video = join(self.work_dir, video_files[0])

        # ------------------------- Keywords --------------------------------------------
        keywords, keysentence = self.get_keywords(video)

        # ---------------- Clean away unwanted candidates generated by HECATE -----------
        img_dir = self.vid_to_frame(video)
        img_list_file = self.cleaning(video, img_dir)
        img_list = sorted([line for line in open(img_list_file, "r").read().splitlines() if line][1:])

        # -------------------------  DO MATCHING ----------------------------------------
        self.out_dir = join(self.work_dir, '{}-{}'.format(
            self.args.method, basename(img_list_file.replace('.txt', ''))))
        # generate captions if needed
        if 'cap' in self.args.method or 'ensemble' in self.args.method:
            caption_file = '{}-captions.txt'.format(img_list_file.replace('.txt', ''))
            if not exists(caption_file) or \
                    int(open(img_list_file, "r").readline().strip()) != len([line for line in open(caption_file, "r").read().splitlines() if line]):
                caption_frames(img_dir, img_list_file, batch_size=self.args.caption_batch, use_fp16=True)

        if self.args.method in ['keysen2img', 'keysen_ensemble']:
            scores1, reasons = keysen_to_imgs(
                keysentence, img_dir, img_list, self.out_dir, self.args.match_batch,
                save_img=True if self.args.method == 'keysen2img' else False
            )

        if self.args.method in ['keysen2cap', 'keysen_ensemble']:
            scores2, reasons = keysen_to_caps(
                keysentence, img_dir, caption_file, self.out_dir,
                save_img=True if self.args.method == 'keysen2cap' else False)

        if self.args.method in ['keys2img', 'keys_ensemble']:
            scores3, reasons = keys_to_imgs(
                keywords, img_dir, img_list, self.out_dir, self.args.match_batch,
                save_img=True if self.args.method == 'keys2img' else False)

        if self.args.method in ['keys2cap', 'keys_ensemble']:
            scores4, reasons = keys_to_caps(
                keywords, img_dir, caption_file, self.out_dir,
                save_img=True if self.args.method == 'keys2cap' else False)

        if self.args.method == 'keysen_ensemble':
            topk = 10
            scores = 0.6 * scores1 + 0.4 * scores2
            topk_inds = np.argpartition(scores, -topk)[-topk:]
            reasons = []
            for i in topk_inds.tolist():
                reason = '\t'.join([img_list[i], 'No caption', keysentence, "{:.4f}".format(scores[i])])
                copy2(join(img_dir, img_list[i]), self.out_dir)
                reasons.append(reason)
            write_reason(reasons, join(self.out_dir, 'reasoning.txt'), keysentence)

        if self.args.method == 'keys_ensemble':
            scores = (scores3 + scores4) / 2
            reasons = filter_doubles(scores, keywords, img_dir, img_list, self.out_dir, save_img=True)
            write_reason(reasons, join(self.out_dir, 'reasoning.txt'), keywords)

        print("STEP3: Matching done.")

        # sort reasons in chronological order
        reasons = sorted(reasons, key=lambda x: int(x.split('\t')[0].split('.')[0]))

        return reasons

    def run(self, any_path):    # process all the videos in the given directory
        # parse directory
        videos = []
        for root, dirs, files in os.walk(any_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_files = [d for d in os.listdir(self.work_dir) if d.endswith('.mp4')]
                    assert len(video_files) == 1, \
                        "Number of video files are not correct for: {}".format(self.work_dir)
                    videos.append(join(root, video_files[0]))
        print("In this run, we are going to process {} videos".format(len(videos)))
        for video in videos:
            self.run_single_video(video)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_batch', type=int, default=32,
                        help='batch size')
    parser.add_argument('--caption_batch', type=int, default=2,
                        help="batch size for caption caption module")
    parser.add_argument('--clean_batch', type=int, default=4,
                        help="batch size for caption caption module")
    parser.add_argument('--din', "-d", type=str, default=None)
    parser.add_argument('--post_hecate', action='store_true',
                        help="use image selected by hecate")
    parser.add_argument('--gap_in_seconds', default=4, type=int)
    parser.add_argument('--method', '-m', default='en',
                        choices=['keysen2img', 'keysen2cap', 'keys2img', 'keys2cap',
                                 'keysen_ensemble', 'keys_ensemble'],
                        )
    parser.add_argument('--aes_path', type=str, default="/raid/P15/4-data/aes")
    parser.add_argument('--meta_path', type=str, default='/raid/P15/4-data/mediacorp/Metadata.xlsx')
    parser.add_argument('--cleaning', type=str, choices=[None, 'c1', 'c2'], default=None,
                        help='c1: remove similar images'
                             'c2: remove similar images + remove opening/ending parts')
    args = parser.parse_args()

    return args


args = get_args()
matcher = SemanticMatch(args)
matcher.run_single_video('/home/john/Desktop/P16/1-data/MediaCorp/mediacorp-test-set/CLIF/DAQ25425.mp4')
