import argparse
import json
import multiprocessing as mp
import os
from functools import partial

import cv2
import dlib
from imutils import face_utils
from tqdm import tqdm


def divide_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def parse_info(infos_and_worker_num, face_path):
    infos, worker_num = infos_and_worker_num
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_81_face_landmarks.dat"
    face_predictor = dlib.shape_predictor(predictor_path)
    pairs = []
    print(f"worker {num_workers} start")
    for info in tqdm(infos, desc=str(worker_num), position=worker_num):
        file = info.split()[0]
        face_file_path = os.path.join(face_path, file)
        face_bgr = cv2.imread(face_file_path, cv2.IMREAD_UNCHANGED)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)  # hw3
        faces = face_detector(face_rgb, 1)
        if len(faces) == 0:
            # print(info, "No Face Detected!!!")
            continue
        # print(info, "Detected multiple faces")
        # print(info, "Detected multiple faces")
        landmark = face_predictor(face_rgb, faces[0])
        landmark = face_utils.shape_to_np(landmark).tolist()
        pairs.append((file, landmark))
    print(f"worker {num_workers} got {len(pairs)} lmks")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="path", type=str, help="The path of faces")
    parser.add_argument("-t", dest="txt", type=str, help="The split txt")
    parser.add_argument(
        "-n", dest="workers", type=int, default=8, help="Number of workers"
    )
    args = parser.parse_args()

    face_path = args.path
    txt_file = args.txt
    print(face_path, txt_file)
    with open(os.path.join(face_path, txt_file), "r") as f:
        infos = [
            line
            for line in f.readlines()
            if os.path.exists(os.path.join(face_path, line.split()[0]))
        ]
    print("Start parsing")
    num_workers = args.workers
    list_of_infos = divide_list(infos, num_workers)
    list_of_pairs = []
    with mp.Pool(num_workers) as workers:
        for pairs in workers.imap_unordered(
            partial(
                parse_info,
                face_path=face_path,
            ),
            zip(list_of_infos, range(num_workers)),
        ):
            list_of_pairs += pairs

    landmarks_dict = {}
    for file, landmark in list_of_pairs:
        landmarks_dict[file] = landmark
    print("Start saving")
    with open(os.path.join(face_path, f"{txt_file[:-4]}_landmarks2.json"), "w") as f:
        json.dump(landmarks_dict, f)
    print("Saving done")
