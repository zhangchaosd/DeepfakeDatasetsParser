import argparse
import multiprocessing as mp
import os
import random
from functools import partial
from multiprocessing import Lock

import cv2
import numpy as np

# use_cuda_decoder = False
use_cuda_decoder = hasattr(cv2, 'cudacodec')
print(f'Use GPU decoder: {use_cuda_decoder}')

# conda install -c https://conda.anaconda.org/conda-forge dlib
import dlib
# from retinaface.pre_trained_models import get_model
from tqdm import tqdm

# device = 'cuda:0'
# device = 'cpu'
# rt_detector = get_model("resnet50_2020-07-20", max_size=1024, device=device)
# rt_detector.eval()

dlib_detector = dlib.get_frontal_face_detector() #获取人脸分类器

mutex = Lock()

SEED = 1021


# no masks
def parse_videos_mp(videos, label, path, save_path, faces_prefix, samples, face_scale, detector, num_workers, log_info, fc):
    print(log_info)
    infos = []
    with mp.Pool(num_workers) as workers:
        with tqdm(total=len(videos)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    fc,
                    label=label,
                    path=path,
                    save_path=save_path,
                    faces_prefix=faces_prefix,
                    samples=samples,
                    face_scale=face_scale,
                    detector=detector,
                ),
                videos,
            ):
                pbar.update()
                infos += info
    print(log_info, len(infos))
    return infos


def parse_video(
    dataset_path,
    rela_path,
    video_name,
    label,
    f,
    samples,
    face_scale,
    detector,
):
    face_save_path = os.path.join(dataset_path, 'faces', rela_path)
    video_path = os.path.join(dataset_path, rela_path, video_name)
    face_names = video2face_pngs(
        video_path, face_save_path, samples, face_scale, detector
    )
    if face_names == None:
        return
    f.writelines(
        [
            os.path.join('faces', rela_path, face_name)
            + '\t'
            + str(label)
            + '\n'
            for face_name in face_names
        ]
    )


def video2face_pngs(video_path, save_path, samples, face_scale, detector):
    gen_dirs(save_path)
    _, file_names, frames = video2frames(video_path, samples)
    if frames == None:
        return None
    faces = [*map(partial(img2face, face_scale = face_scale, detector = detector), frames)]
    file_names = [file_name for i, file_name in enumerate(file_names) if faces[i] is not None]
    faces = [face for face in faces if face is not None]
    [
        *map(
            cv2.imwrite,
            [os.path.join(save_path, fn) for fn in file_names],
            faces,
        )
    ]
    return file_names


def video2frames(video_path, samples):
    if not os.path.exists(video_path):
        print(f'Video file not exists! {video_path}')
        return [],[],[]
    frames = []
    if use_cuda_decoder:
        cap=cv2.cudacodec.createVideoReader(video_path)
        ret, frame = cap.nextFrame()
        while ret:
            frames.append(frame)
            ret, frame = cap.nextFrame()
        frame_count = len(frames)
        samples = min(samples, frame_count)
        stride = frame_count // samples + 1
        order = [i for i in range(frame_count) if i % stride == 0][:samples]
        frames = [frame.download()[:,:,:3] for i, frame in enumerate(frames) if i in order]
    else:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            print(f'Video has not frame! {video_path}')
            return [],[],[]
        frame_idxs = np.linspace(0, frame_count - 1, samples, dtype=np.int32)
        new_frame_idxs = []
        for num in range(frame_count):
            flag, frame = cap.read()  # bgr
            if (not flag) or (num not in frame_idxs):
                continue
            new_frame_idxs.append(num)
            frames.append(frame)
        order = new_frame_idxs
        cap.release()
    video_name = os.path.basename(video_path)
    file_names = [video_name[:-4] + f'_{i}.png' for i, _ in enumerate(order)]
    assert len(file_names) == len(frames)
    return order, file_names, frames


def img2face(img, face_scale, detector):
    crop_data = get_face_location(img, face_scale, detector)
    if crop_data is None:
        return None
    img = crop_img(img, crop_data)
    return img


def get_face_location(img, face_scale, detector):
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img: h, w, c rgb
    if detector == 'dlib':  #  TODO
        annotation = dlib_detector(img, 1) # h, w, c rgb
        if len(annotation)==0:
            return None
        x1 = annotation[0].left()
        y1 = annotation[0].top()
        x2 = annotation[0].right()
        y2 = annotation[0].bottom()
    else:
        annotation = rt_detector.predict_jsons(img, confidence_threshold=0.3)
        if len(annotation[0]['bbox']) == 0:
            return None
        x1, y1, x2, y2 = annotation[0]['bbox']
        # x1, y1, x2, y2 = 0,0,20,20
    x1, y1, x2, y2 = list(
        map(
            int,
            [
                x1 - (x2 - x1) * (face_scale - 1) / 2,
                y1 - (y2 - y1) * (face_scale - 1) / 2,
                x2 + (x2 - x1) * (face_scale - 1) / 2,
                y2 + (y2 - y1) * (face_scale - 1) / 2,
            ],
        )
    )
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return y1, y2, x1, x2


def crop_img(img, xy):
    min_y, max_y, min_x, max_x = xy
    if len(img.shape)==3:
        return img[min_y:max_y, min_x:max_x, :]
    elif len(img.shape)==2:
        return img[min_y:max_y, min_x:max_x]
    else:
        assert False


def get_files_from_path(path):
    files = os.listdir(path)
    return [f for f in files if f != '.DS_Store' and f != '.dircksum']

def read_txt(path):
    assert path.endswith('.txt')
    with open(path) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def gen_dirs(path):
    mutex.acquire()
    if not os.path.exists(path):
        os.makedirs(path)
    mutex.release()


def static_shuffle(l):
    l.sort()
    random.seed(SEED)
    random.shuffle(l)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', required=True, help='The path of be dataset.')
    parser.add_argument('-save_path', default='', help='The path of be faces to be saved.')
    parser.add_argument(
        '-samples',
        default=8,
        type=int,
        help='Number of frames acquired from each video. Default is 8',
    )
    parser.add_argument(
        '-scale',
        default=1.3,
        type=float,
        help='Crop scale rate of face bbox. Default is 1.3',
    )
    parser.add_argument(
        '-subset',
        default='FF',
        type=str,
        help='FF or DFD(DeepFakeDetection), only avaliable for FaceForensics++.',
    )
    parser.add_argument(
        '-detector',
        default='dlib',
        type=str,
        help='dlib or retinaface.',
    )
    parser.add_argument(
        '-workers',
        default=1,
        type=int,
        help='Number of processes.',
    )
    parser.add_argument(
        '-part',
        default=0,
        type=int,
        help='Number of part',
    )
    parser.add_argument(
        '-mode',
        default='train',
        type=str,
        help='For DFDC. train or test or validation',
    )
    parser.add_argument(
        '-path_auth',
        default='VidTIMIT_path',
        type=str,
        help='For DeepfakeTIMIT. The path of VidTIMIT.',
    )
    args = parser.parse_args()
    if args.save_path == '':
        args.save_path = args.path
    return args
