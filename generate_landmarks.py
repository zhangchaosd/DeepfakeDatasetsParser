import argparse
import json
import os
import sys

import cv2
import dlib
from imutils import face_utils
from tqdm import tqdm


def parse_info(face_path, landmarks_dict, face_detector, face_predictor, info):
    file = info.split()[0]
    face_file_path = os.path.join(face_path, file)
    face_bgr = cv2.imread(face_file_path, cv2.IMREAD_UNCHANGED)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)  # hw3
    faces = face_detector(face_rgb, 1)
    if len(faces) == 0:
        print(info, "No Face Detected!!!")
        return
    elif len(faces) > 1:
        print(info, "Detected multiple faces")
    landmark = face_predictor(face_rgb, faces[0])
    landmark = face_utils.shape_to_np(landmark).reshape(81,2).tolist()
    landmarks_dict[file] = landmark


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
    parser.add_argument('-p', target='path', type=str, help='The path of faces')
    parser.add_argument('-t', target='txt', type=str, help='The split txt')
    args = parser.parse_args()

    face_path = args.path
    txt_file = args.txt
    with open(os.path.join(face_path, txt_file), 'r') as f:
        infos = [line for line in f.readlines() if os.path.exists(os.path.join(face_path, line.split()[0]))]
    landmarks_dict = {}
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    for info in tqdm(infos):
        parse_info(face_path, landmarks_dict, face_detector, face_predictor, info)
    with open(os.path.join(face_path, f"{txt_file[:-4]}_landmarks.json"), 'w') as f:
        json.dump(landmarks_dict, f)
