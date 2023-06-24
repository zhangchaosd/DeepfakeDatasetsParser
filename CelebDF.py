import concurrent.futures
import json
import os
import sys
from glob import glob

from tqdm import tqdm

from utils2 import parse_video


def worker(args):
    dataset_path, video_path, num_frames_to_extract, label = args
    return parse_video(dataset_path, video_path, num_frames_to_extract, label)


def parse_split(dataset_path, video_infos, num_frames_to_extract, split):
    print(f"Now start parsing {split}: {len(video_infos)}")
    split_infos = []
    split_landmarks = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for infos, landmarks in tqdm(
            executor.map(
                worker,
                [
                    (dataset_path, video_path, num_frames_to_extract, label)
                    for video_path, label in video_infos
                ],
            ),
            total=len(video_infos),
        ):
            split_infos.extend(infos)
            split_landmarks.update(landmarks)
    split_info_txt = os.path.join(
        dataset_path, f"faces_{num_frames_to_extract}", f"{split}.txt"
    )
    with open(split_info_txt, "w") as f:
        f.writelines(split_infos)
    print(f"{split}: ", len(split_infos))
    landmarks_path = split_info_txt[:-3] + "json"
    with open(landmarks_path, "w") as f:
        json.dump(split_landmarks, f)
    return split_infos, split_landmarks


def main(dataset_path, num_frames_to_extract):
    video_infos = (
        [
            (os.path.relpath(video_path, dataset_path), 0)
            for video_path in glob(os.path.join(dataset_path, "Celeb-real/*.mp4"))
        ]
        + [
            (os.path.relpath(video_path, dataset_path), 0)
            for video_path in glob(os.path.join(dataset_path, "YouTube-real/*.mp4"))
        ]
        + [
            (os.path.relpath(video_path, dataset_path), 1)
            for video_path in glob(os.path.join(dataset_path, "Celeb-synthesis/*.mp4"))
        ]
    )

    test_txt_path = os.path.join(dataset_path, "List_of_testing_videos.txt")
    with open(test_txt_path, "r") as f:
        test_videos = [video.split()[1] for video in f.read().splitlines()]
    train_split = [info for info in video_infos if info[0] not in test_videos]
    test_split = [info for info in video_infos if info[0] in test_videos]

    all_infos = []
    all_landmarks = {}
    train_infos, landmarks = parse_split(
        dataset_path, train_split, num_frames_to_extract, "train"
    )
    all_infos.extend(train_infos)
    all_landmarks.update(landmarks)
    test_infos, landmarks = parse_split(
        dataset_path, test_split, num_frames_to_extract, "test"
    )
    all_infos.extend(test_infos)
    all_landmarks.update(landmarks)

    all_txt_path = os.path.join(
        dataset_path, f"faces_{num_frames_to_extract}", "all.txt"
    )
    with open(all_txt_path, "w") as f:
        f.writelines(all_infos)
    print("Total: ", len(all_infos))
    landmarks_path = all_txt_path[:-3] + "json"
    with open(landmarks_path, "w") as f:
        json.dump(all_landmarks, f)


# 0 is real
# python CelebDF2.py '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF-v2' 0
# python CelebDF2.py '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF-v2' 0
if __name__ == "__main__":
    dataset_path = sys.argv[1]
    num_frames_to_extract = int(sys.argv[2])
    main(dataset_path, num_frames_to_extract)
