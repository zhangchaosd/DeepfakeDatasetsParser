import concurrent.futures
import csv
import json
import os
import sys

from tqdm import tqdm

from utils2 import parse_video

"""
#!/usr/bin/env bash

for i in {0..49}
do
    unzip dfdc_train_part_${i}.zip
done

!!!!!!! important !!!!!!!
jar xvf dfdc_train_part_23.zip
"""


def worker(args):
    dataset_path, video_path, num_frames_to_extract, label = args
    return parse_video(dataset_path, video_path, num_frames_to_extract, label)


lb = {
    "FAKE": "1",
    "REAL": "0",
}


def get_infos_from_path(dataset_path, rela_path):
    videos = []
    if rela_path.startswith("val"):
        with open(
            os.dataset_path.join(dataset_path, rela_path, "labels.csv"), "r"
        ) as f_csv:
            csv_reader = csv.reader(f_csv)
            csv_reader.__next__()
            for video, label in tqdm(csv_reader):
                videos.append((os.path.join(rela_path, video), label))
    else:
        with open(os.path.join(dataset_path, rela_path, "metadata.json")) as f_json:
            metadata = json.load(f_json)
            for video in metadata.keys():
                if rela_path.startswith("tr"):
                    videos.append(
                        (os.path.join(rela_path, video), lb[metadata[video]["label"]])
                    )
                else:
                    videos.append(
                        (os.path.join(rela_path, video), metadata[video]["is_fake"])
                    )
    return videos


def get_split_video_infos(dataset_path, split):
    video_infos = []
    if split == "train":
        for i in range(50):
            video_infos += get_infos_from_path(
                dataset_path, os.path.join(split, f"dfdc_train_part_{i}")
            )
    else:
        video_infos = get_infos_from_path(dataset_path, split)
    return video_infos


def main(dataset_path, num_frames_to_extract):
    splits = ["train", "validation", "test"]
    all_infos = []
    all_landmarks = {}
    for split in splits:
        print(f"Parsing {split}...")
        split_infos = []
        split_landmarks = {}
        video_infos = get_split_video_infos(dataset_path, split)
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
        all_infos.extend(split_infos)
        all_landmarks.update(split_landmarks)
        print(f"Num of {split}: ", len(split_infos))
    all_txt_path = os.path.join(
        dataset_path, f"faces_{num_frames_to_extract}", "all.txt"
    )
    with open(all_txt_path, "w") as f:
        f.writelines(all_infos)
    landmarks_path = all_txt_path[:-3] + "json"
    with open(landmarks_path, "w") as f:
        json.dump(all_landmarks, f)
    print("Total: ", len(all_infos))


#  1 is fake
# python DFDC2.py '/share/home/zhangchao/datasets_io03_ssd/DFDC' 3
if __name__ == "__main__":
    dataset_path = sys.argv[1]
    num_frames_to_extract = int(sys.argv[2])
    main(dataset_path, num_frames_to_extract)
