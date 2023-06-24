import os

import cv2
import dlib  # conda install -c https://conda.anaconda.org/conda-forge dlib
import numpy as np
from imutils import face_utils

# from retinaface.pre_trained_models import get_model
# device = 'cuda:0'
# device = 'cpu'
# rt_detector = get_model("resnet50_2020-07-20", max_size=1024, device=device)
# rt_detector.eval()

SEED = 1021
face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_81_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)


def clip_value(value, lower_bound, upper_bound):
    return max(min(value, upper_bound), lower_bound)


def parse_image(img, save_path, face_scale=1.3):
    if isinstance(img, str):
        assert os.path.exists(img)
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        assert len(img.shape) == 3
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector(img, 1)  # h, w, c rgb
    if len(faces) == 0:
        return None
    landmark = face_predictor(img, faces[0])
    landmark = face_utils.shape_to_np(landmark).tolist()
    x1 = faces[0].left()
    y1 = faces[0].top()
    x2 = faces[0].right()
    y2 = faces[0].bottom()
    x1, y1, x2, y2 = [
        int(x1 - (x2 - x1) * (face_scale - 1) / 2),
        int(y1 - (y2 - y1) * (face_scale - 1) / 2),
        int(x2 + (x2 - x1) * (face_scale - 1) / 2),
        int(y2 + (y2 - y1) * (face_scale - 1) / 2),
    ]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    face = img[y1:y2, x1:x2, :]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    landmark = [[x - x1, y - y1] for x, y in landmark]
    h, w, c = face.shape
    landmark = [[clip_value(x, 0, w - 1), clip_value(y, 0, h - 1)] for x, y in landmark]
    ######  TODO del
    # for x, y in landmark:
    # face[y][x] = [255, 255, 255]
    ######
    cv2.imwrite(save_path, face)
    return landmark


def parse_video(dataset_path, rela_path, num_frames_to_extract, label):
    assert num_frames_to_extract >= 0
    video_path = os.path.join(dataset_path, rela_path)
    assert os.path.exists(video_path)
    output_path = os.path.join(
        dataset_path, f"faces_{num_frames_to_extract}", rela_path[:-4]
    )
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"Video has not frame! {video_path}")
        return [], [], []
    num_frames_to_extract = min(num_frames_to_extract, frame_count)
    if num_frames_to_extract > 0:
        frame_idxs = np.linspace(
            0, frame_count - 1, num_frames_to_extract, dtype=np.int32
        )
    else:
        frame_idxs = list(range(frame_count))
    infos = []
    landmarks = {}
    for num in range(frame_count):
        flag, frame = cap.read()  # bgr
        if (not flag) or (num not in frame_idxs):
            continue
        face_path = os.path.join(rela_path[:-4], f"{num}.png")
        landmark = parse_image(
            frame,
            os.path.join(dataset_path, f"faces_{num_frames_to_extract}", face_path),
        )
        if landmark:
            infos.append(face_path + "\t" + str(label) + "\n")
            landmarks[face_path] = landmark
    cap.release()
    return infos, landmarks


# infos, landmarks = parse_video(
#     "/share/home/zhangchao/datasets_io03_ssd/UADFV", "fake/0000_fake.mp4", 3, 1
# )
# print(infos[0])
# print(landmarks)
