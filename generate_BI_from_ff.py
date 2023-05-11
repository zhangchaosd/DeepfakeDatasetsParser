import argparse
import json
import multiprocessing as mp
import os
from functools import partial
from os.path import join

import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm

DLIB_MODEL_PATH = "/Users/zhangchao/Downloads/shape_predictor_68_face_landmarks0.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
is_debug = True


def get_dlib_face(img):
    """
    调用dlib获取人脸位置
    :param img: 待截取人脸的图片
    :return: 人脸位置对象face，包括(face.left(), face.top()), (face.right(), face.bottom())
    """
    faces = detector(img, 0)
    if len(faces) == 0:
        return None
    else:
        return faces[0]


def get_face_landmarks(img, face):
    """
    获取图片中的人脸坐标，FF++数据集大多数人脸只有一个，且在正面
    :param img: 图片
    :param face: dlib获取的人脸框位置
    :return: 人脸68特征点坐标，形状为(68,2)，格式为numpy数组
    """
    shape = predictor(img, face)
    # 将dlib检测到的人脸特征点转为numpy格式
    res = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        # 68个特征点
        res[i] = np.array([shape.part(i).x, shape.part(i).y])

    return res


def conservative_crop(img, face, scale=1.3, new_size=(317, 317)):
    """
    FF++论文中裁剪人脸是先找到人脸框，然后人脸框按比例扩大后裁下更大区域的人脸
    :param img: 待裁剪人脸图片
    :param face: dlib获取的人脸区域
    :param scale: 扩大比例
    :return: 裁剪下来的人脸区域，大小默认为(256,256)，Implementation detail中说预测的mask上采样为256*256，所以截取的人脸应该也是这个大小
    """

    height, width = img.shape[:2]

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    cropped = img[y1 : y1 + size_bb, x1 : x1 + size_bb, :]
    # cropped = cv2.resize(img[y1 : y1 + size_bb, x1 : x1 + size_bb, :], new_size)

    return cropped, x1, y1


def extract_frames(data_path, frame_num=120):
    video_number = int(os.path.basename(data_path)[:3])
    reader = cv2.VideoCapture(data_path)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indexs = np.linspace(
        0, frame_count - 1, min(frame_num, frame_count), dtype=int
    )
    faces_lms = []
    for frame_index in range(0, frame_indexs[-1] + 1):
        _, image = reader.read()
        if frame_index not in frame_indexs:
            continue
        face = get_dlib_face(image)
        if face is None:
            print("found no face in ", data_path)
            continue
        lms = get_face_landmarks(image, face)
        image, crop_left, crop_top = conservative_crop(image, face)
        lms = [[lm[0] - crop_left, lm[1] - crop_top] for lm in lms]
        assert len(lms) == 68
        faces_lms.append((image, lms, video_number))
    reader.release()
    return faces_lms


def get_train_index(split_dir):
    train_idx = []
    with open(os.path.join(split_dir, "train.json")) as f:
        load_dict = json.load(f)
        for pair in load_dict:
            train_idx.append(pair[0])
            train_idx.append(pair[1])
    return train_idx


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集根目录
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="/Users/zhangchao/Downloads/ff_or",
        help="FF++ dataset root path",
    )
    # 提取人脸输出文件夹
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="/Users/zhangchao/Downloads/ff_or/faces_train_lmks",
        help="output path",
    )
    # 训练、验证、测试切分文件夹
    parser.add_argument(
        "-s",
        "--split_dir",
        type=str,
        default="/Users/zhangchao/Downloads/ff_or/splits",
        help="directory of the split json file",
    )
    # 数据压缩
    parser.add_argument(
        "-c", "--compression", type=str, default="raw", help="compression of video"
    )

    parser.add_argument(
        "-n",
        "--frames",
        type=int,
        default=120,
        help="extract frames number in one video",
    )

    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=8,
        help="number of processes to extract frames",
    )

    parser.add_argument(
        "-p",
        "--parts",
        type=int,
        default=10,
        help="number of parts to extract frames (memory limited)",
    )
    args = parser.parse_args()
    return args


def main(args, iteration):
    print(iteration)
    train_idx = get_train_index(args.split_dir)
    root_path = args.data_path
    dataset_path = "original_sequences/youtube"
    videos_path = join(root_path, dataset_path, "raw", "videos")
    faces_lmks = []

    videos = [
        join(videos_path, video)
        for video in os.listdir(videos_path)
        if video[:3] in train_idx
    ]
    assert len(videos) == 720
    videos = sorted(videos, key=lambda x: int(x.split("/")[-1][:3]))[
        iteration * 72 : iteration * 72 + 72
    ]
    with mp.Pool(args.num_workers) as workers:
        with tqdm(total=len(videos)) as pbar:
            for info in workers.imap_unordered(
                partial(extract_frames, frame_num=args.frames),
                videos,
            ):
                pbar.update()
                faces_lmks.extend(info)

    return faces_lmks


if __name__ == "__main__":
    args = parse_args()
    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    start_index = 0
    lmks_video_number = []
    for i in range(args.parts):
        print(f"Start parse part{i}")
        faces_lmks = main(args, i)
        for info in faces_lmks:
            torch.save(info[0], join(save_path, f"{start_index}.pyt"))
            start_index += 1
            lmks_video_number.append((info[1], info[2]))
    torch.save(lmks_video_number, join(save_path, "lmks_video_number.pyth"))
    print(len(lmks_video_number))  # 86356
    print("All done")
