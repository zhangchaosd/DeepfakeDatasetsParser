import multiprocessing as mp
import os
from functools import partial

import cv2
from tqdm import tqdm

from utils import (
    crop_img,
    gen_dirs,
    get_face_location,
    get_files_from_path,
    parse,
    static_shuffle,
    video2frames,
    video2face_pngs
)


def parse_split(
    train_subjects,
    faces_path,
    hq_path,
    lq_path,
    f_hq,
    f_lq,
    label,
    samples,
    face_scale,
):
    for subject in tqdm(train_subjects):
        videos = [
            f
            for f in os.listdir(os.path.join(hq_path, subject))
            if f.endswith('.avi')
        ]
        assert len(videos) == 10
        for video in videos:
            gen_dirs(os.path.join(faces_path, 'higher_quality', subject))
            gen_dirs(os.path.join(faces_path, 'lower_quality', subject))
            # hq
            file_names, frames = video2frames(
                os.path.join(hq_path, subject, video), samples
            )
            crop_datas = [
                *map(get_face_location, frames, [face_scale] * len(frames))
            ]
            faces = [*map(crop_img, frames, crop_datas)]
            [
                *map(
                    cv2.imwrite,
                    [
                        os.path.join(
                            faces_path, 'higher_quality', subject, img_name
                        )
                        for img_name in file_names
                    ],
                    faces,
                )
            ]
            f_hq.writelines(
                [
                    os.path.join('faces', 'higher_quality', subject, img_name)
                    + '\t'
                    + label
                    + '\n'
                    for img_name in file_names
                ]
            )

            # lq
            _, frames = video2frames(
                os.path.join(lq_path, subject, video), samples
            )
            faces = [*map(crop_img, frames, crop_datas)]
            [
                *map(
                    cv2.imwrite,
                    [
                        os.path.join(
                            faces_path, 'lower_quality', subject, img_name
                        )
                        for img_name in file_names
                    ],
                    faces,
                )
            ]
            f_lq.writelines(
                [
                    os.path.join('faces', 'lower_quality', subject, img_name)
                    + '\t'
                    + label
                    + '\n'
                    for img_name in file_names
                ]
            )

def f3(video_path, path, faces_prefix, samples, face_scale, detector):
    rela_path = os.path.dirname(video_path)
    save_path = os.path.join(path, faces_prefix, rela_path)
    video_path = os.path.join(path, video_path)
    infos = [
        os.path.join(faces_prefix, rela_path, img_name) + '\t1\n'
        for img_name in video2face_pngs(
            video_path, save_path, samples, face_scale, detector
        )
    ]
    return infos

def f2(mode, split, path, faces_prefix, samples, face_scale, detector, num_workers):
    videos = []
    for subject in split:
        videos += [os.path.join(subject, file) for file in get_files_from_path(os.path.join(path, subject)) if file.endswith('.avi')]
    
    txt_path = os.path.join(path, faces_prefix, mode + '.txt')
    infos = []
    with mp.Pool(num_workers) as workers, open(txt_path, 'w') as f:
        with tqdm(total=len(split)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    f3,
                    path=path,
                    faces_prefix=faces_prefix,
                    samples=samples,
                    face_scale=face_scale,
                    detector=detector,
                ),
                videos,
            ):
                pbar.update()
                infos += info
        f.writelines(infos)
    print(mode, len(infos))

def f1(quality, path, faces_prefix, samples, face_scale, detector, num_workers):
    gen_dirs(os.path.join(path, faces_prefix, quality))
    # subjects = get_files_from_path(os.path.join(path, faces_prefix, quality))
    subjects = [os.path.join(quality, subject) for subject in get_files_from_path(os.path.join(path, faces_prefix, quality))]
    assert len(subjects) == 32
    static_shuffle(subjects)
    train_split = subjects[:22]
    test_split = subjects[22:]
    all_info = []
    all_info += f2('train_'+quality, train_split, path, faces_prefix, samples, face_scale, detector, num_workers)
    all_info += f2('test_'+quality, test_split, path, faces_prefix, samples, face_scale, detector, num_workers)
    with open(os.path.join(path, faces_prefix, 'all_'+quality+'.txt'), 'w') as f:
        f.writelines(all_info)
    print('All ', len(all_info))

def main(path, samples, face_scale, detector, num_workers):
    faces_prefix = 'faces' + str(samples) + detector
    gen_dirs(os.path.join(path, faces_prefix))
    f1('lower_quality', path, faces_prefix, samples, face_scale, detector, num_workers)
    f1('higher_quality', path, faces_prefix, samples, face_scale, detector, num_workers)
    train_hq_txt = os.path.join(faces_path, 'train_hq.txt')
    test_hq_txt = os.path.join(faces_path, 'test_hq.txt')
    train_lq_txt = os.path.join(faces_path, 'train_lq.txt')
    test_lq_txt = os.path.join(faces_path, 'test_lq.txt')
    f_train_hq = open(train_hq_txt, 'w')
    f_test_hq = open(test_hq_txt, 'w')
    f_train_lq = open(train_lq_txt, 'w')
    f_test_lq = open(test_lq_txt, 'w')

    hq_path = os.path.join(path, 'higher_quality')
    lq_path = os.path.join(path, 'lower_quality')
    subjects = get_files_from_path(hq_path)
    assert len(subjects) == 32
    static_shuffle(subjects)
    train_subjects = subjects[:22]
    test_subjects = subjects[22:]
    label = '1'

    print('Parsing train split...')
    parse_split(
        train_subjects,
        faces_path,
        hq_path,
        lq_path,
        f_train_hq,
        f_train_lq,
        label,
        samples,
        face_scale,
    )

    print('Parsing test split...')
    parse_split(
        test_subjects,
        faces_path,
        hq_path,
        lq_path,
        f_test_hq,
        f_test_lq,
        label,
        samples,
        face_scale,
    )

    f_train_hq.close()
    f_test_hq.close()
    f_train_lq.close()
    f_test_lq.close()


# 1 is fake
if __name__ == '__main__':
    args = parse()
    # main('/share/home/zhangchao/datasets_io03_ssd/DeepfakeTIMIT', 100, 1.3)
    main(args.path, args.samples, args.scale, args.detector, args.workers)
