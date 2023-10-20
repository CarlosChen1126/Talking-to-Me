import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from time import time

cnt_video = 1

for whole_vidio_id in glob.glob('./videos/*'):

    if cnt_video < 31:

        video_id = whole_vidio_id.split('/')[-1]
        os.mkdir(f'./split_frame/{video_id[:-4]}')
        cap = cv2.VideoCapture(os.path.join('./videos', video_id))

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in tqdm( range(num_frames) ):
            ret, frame = cap.read()
            cv2.imwrite(os.path.join(f'./split_frame/{video_id[:-4]}', f'frame_{idx}.png'), frame)

            raise

        print(f'finish {cnt_video} !')
        cnt_video += 1