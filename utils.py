import argparse
from functools import partial
from multiprocessing import Lock
import os
import random

import cv2
# conda install -c https://conda.anaconda.org/conda-forge dlib
import dlib
# from retinaface.pre_trained_models import get_model

# device = 'cuda:0'
# device = 'cpu'
# rt_detector = get_model("resnet50_2020-07-20", max_size=1024, device=device)
# rt_detector.eval()

dlib_detector = dlib.get_frontal_face_detector() #获取人脸分类器

mutex = Lock()

SEED = 1021


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


def video2frames(video_path, samples, order = None):
    if not os.path.exists(video_path):
        print(f'Video file not exists! {video_path}')
        return None
    frames = []
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames == 0:
        print(f'Video has not frame! {video_path}')
        return None
    samples = min(samples, num_frames)
    stride = num_frames // samples + 1
    if order == None:
        order = [i for i in range(num_frames) if i % stride == 0][:samples]
    new_order = []
    for num in order:
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        flag, frame = cap.read()  # bgr
        if not flag:
            continue
        new_order.append(num)
        frames.append(frame)
    order = new_order
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
    return img[min_y:max_y, min_x:max_x, :]


def get_files_from_path(path):
    files = os.listdir(path)
    return [f for f in files if f != '.DS_Store' and f != '.dircksum']

def read_txt(path):
    assert path.endswith('.txt')
    lines = []
    with open(path) as f:
        for line in f.readlines():
            lines.append(line.strip())
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
    args = parser.parse_args()
    return args
