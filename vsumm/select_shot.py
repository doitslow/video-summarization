import os
import cv2
import glob
import shutil
import numpy as np
from os.path import join, dirname, basename, exists
from text2frame.content_pipe import SemanticMatch, get_args
from shot_detection.inference.transnetv2 import TransNetV2
import subprocess


class Video2Clip(object):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def get_fps(video_path):
        vid_capture = cv2.VideoCapture(video_path)
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        else:
            origin_fps = vid_capture.get(5)
            print('Frames per second : ', origin_fps, 'FPS')
            frame_count = vid_capture.get(7)
            print('Frame count : ', frame_count)

        return origin_fps, frame_count

    @staticmethod
    def sec2time(sec):
        h = int(sec // 3600)
        m = int(sec // 60)
        s = int(sec - h * 3600 - m // 60)

        return "{:02d}:{:02d}:{:02d}".format(h, m, s)

    def draw_text(self, video, text):
        if 'keysen' in self.args.method:
            text = text.strip().split(' ')
            text_split = [text[i*7:(i+1)*7] if i < len(text) // 7 else text[i*7:]
                          for i in range(len(text)//7 + 1)]
            text_split = [' '.join(t) for t in text_split]
            text_split.insert(0, 'Pseudo key sentence matched for this shot: ')
            text_split[1] = '<{}'.format(text_split[1])
            text_split[-1] = '{}>'.format(text_split[-1])
        else:
            text_split = ['<{}>'.format(t.strip()) for t in text.split(';')]
            text_split.insert(0, 'Keywords matched for this shot: ')

        print(text_split)

        with open(video.replace('.mp4', '.txt'), 'w') as fopen:
            for t in text_split:
                fopen.write(t + '\n')
            fopen.close()

        formatting = {
            'fontfile': 'times.ttf',
            'fontcolor': 'red',
            'fontsize': '44',
            'box': '1',
            'boxborderw': '5',
            'boxcolor': 'white@0.75',
            # box = 1:boxborderw = 5:boxcolor = white @ 0.25
            # 'x': '(w-text_w)/2',
            # 'y': '(h-text_h)/2',
            'x': '20',
            'y': '20',
            'fix_bounds': 'True'
        }
        draw = "drawtext=textfile='{}':".format(video.replace('.mp4', '.txt')) \
               + ":".join(['{}={}'.format(k, v) for k, v in formatting.items()])
        command = "/usr/bin/ffmpeg -i {} -vf \"{}\" -c:v libx264 -crf 18 -preset slow -c:a copy {}.mp4"\
            .format(video, draw, video.replace('.mp4', '-with_reason'))
        subprocess.call(command, shell=True)

    def run_single_video(self, video):
        fps, _ = self.get_fps(video)

        # rely on text to frame matching module to set the directories
        text2frame_matcher = SemanticMatch(args)
        selection_with_reasons = text2frame_matcher.run_single_video(video)
        # np.savetxt(join(out_dir, basename(video) + '.selected_frames.txt'),
        #                 selected_frames, fmt="%d")
        selected_frames = [i.split('\t')[0] for i in selection_with_reasons]
        selected_reasons = [i.split('\t')[2] for i in selection_with_reasons]

        fsave = join(text2frame_matcher.temp_dir, basename(video) + '.shots.txt')
        if not exists(fsave):
            shot_detector = TransNetV2()
            video_frames, single_frame_predictions, all_frame_predictions = \
                shot_detector.predict_video(video)
            shots = shot_detector.predictions_to_scenes(single_frame_predictions)
            np.savetxt(fsave, shots, fmt="%d")
        else:
            shots = np.loadtxt(fsave, dtype=int)

        selected_shots = []
        final_selected_reasons = []
        for frame, reason in zip(selected_frames, selected_reasons):
            for shot in shots:
                shot = shot.tolist()
                if shot[0] <= int(frame.replace('.jpeg', '')) <= shot[1] \
                        and (not shot in selected_shots):
                    selected_shots.append(shot)
                    final_selected_reasons.append(reason)

        temp_dir = join(text2frame_matcher.temp_dir, 'temp_clips')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        selected_times = []
        for shot in selected_shots:
            [start_time, end_time] = [f/fps for f in shot]
            if start_time != end_time:
                selected_times.append([start_time, end_time])
        for i, (t, r) in enumerate(zip(selected_times, final_selected_reasons)):
            if not os.path.exists("{}/{}.mp4".format(temp_dir, i)):
                trim_command = "/usr/bin/ffmpeg -ss {} -to {} -i {} -c copy {}/{}.mp4"\
                    .format(t[0], t[1], video, temp_dir, i)
                subprocess.call(trim_command, shell=True)
            self.draw_text('{}/{}.mp4'.format(temp_dir, i), r)

        with open('{}/video_list.txt'.format(temp_dir), 'w') as fopen:
            for f in os.listdir(temp_dir):
                if f.endswith('-with_reason.mp4'):
                    fopen.write("file {}".format(join(temp_dir, f)) + '\n')
            fopen.close()

        merge_command = "/usr/bin/ffmpeg -f concat -safe 0 -i {}/video_list.txt -c copy {}/{}-{}-final_clip.mp4"\
            .format(temp_dir, text2frame_matcher.out_dir, self.args.method, basename(video).replace('.mp4', ''))
        subprocess.call(merge_command, shell=True)
        shutil.rmtree(temp_dir)

    def run(self, any_path):    # process all the videos in the given directory
        # parse directory
        videos = []
        for root, dirs, files in os.walk(any_path):
            for file in files:
                if file.endswith('.mp4') and ('-final_clip' not in file):
                    video_files = [d for d in os.listdir(root) if d.endswith('.mp4')]
                    # assert len(video_files) == 1, "Number of video files are not correct for: {}".format(root)
                    if len(video_files) != 1:
                        print("Number of video files are not correct for: {}".format(root))
                        pass
                    else:
                        videos.append(join(root, video_files[0]))
        print("In this run, we are going to process {} videos".format(len(videos)))
        for video in videos:
            self.run_single_video(video)


if __name__ == "__main__":
    args = get_args()
    video2clip = Video2Clip(args)
    video2clip.run(args.din)