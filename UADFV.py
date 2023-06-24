import concurrent.futures
import json
import os
import sys
from glob import glob

from tqdm import tqdm

from utils2 import parse_video


def handle_video(args):
    dataset_path, video_path, num_frames_to_extract, label = args
    return parse_video(dataset_path, video_path, num_frames_to_extract, label)


def main(dataset_path, num_frames_to_extract):
    all_infos = []
    all_landmarks = {}
    video_paths = [
        (os.path.relpath(video_path, dataset_path), 0)
        for video_path in glob(os.path.join(dataset_path, "real/*.mp4"))
    ] + [
        (os.path.relpath(video_path, dataset_path), 1)
        for video_path in glob(os.path.join(dataset_path, "fake/*.mp4"))
    ]
    assert len(video_paths) == 49 * 2
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for infos, landmarks in tqdm(
            executor.map(
                handle_video,
                [
                    (dataset_path, video_path, num_frames_to_extract, label)
                    for video_path, label in video_paths
                ],
            ),
            total=len(video_paths),
        ):
            all_infos.extend(infos)
            all_landmarks.update(landmarks)
    all_txt_path = os.path.join(
        dataset_path, f"faces_{num_frames_to_extract}", "all.txt"
    )
    with open(all_txt_path, "w") as f:
        f.writelines(all_infos)
    # print("Num of train: ", len(train_info))
    # print("Num of val: ", len(val_info))
    # print("Num of test: ", len(test_info))
    print("Total: ", len(all_infos))
    landmarks_path = all_txt_path[:-3] + "js"
    with open(landmarks_path, "w") as f:
        json.dump(all_landmarks, f)


# 0 is real
# python UADFV2.py '/share/home/zhangchao/datasets_io03_ssd/UADFV' 3
if __name__ == "__main__":
    dataset_path = sys.argv[1]
    num_frames_to_extract = int(sys.argv[2])
    main(dataset_path, num_frames_to_extract)
