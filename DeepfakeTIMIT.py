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
        with tqdm(total=len(videos)) as pbar:
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
    return infos

def f1(quality, path, faces_prefix, samples, face_scale, detector, num_workers):
    gen_dirs(os.path.join(path, faces_prefix, quality))
    # subjects = get_files_from_path(os.path.join(path, faces_prefix, quality))
    subjects = [os.path.join(quality, subject) for subject in get_files_from_path(os.path.join(path, quality))]
    assert len(subjects) == 32
    static_shuffle(subjects)
    train_split = subjects[:22]
    test_split = subjects[22:]
    all_info = []
    print('Parsing train split...')
    all_info += f2('train_'+quality, train_split, path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Parsing test split...')
    all_info += f2('test_'+quality, test_split, path, faces_prefix, samples, face_scale, detector, num_workers)
    with open(os.path.join(path, faces_prefix, 'all_'+quality+'.txt'), 'w') as f:
        f.writelines(all_info)
    print('All ', len(all_info))

def main(path, samples, face_scale, detector, num_workers):
    faces_prefix = 'faces' + str(samples) + detector
    gen_dirs(os.path.join(path, faces_prefix))
    print('Parsing lower_quality...')
    f1('lower_quality', path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Parsing higher_quality...')
    f1('higher_quality', path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Done')


# 1 is fake
# python DeepfakeTIMIT.py -path '/share/home/zhangchao/datasets_io03_ssd/DeepfakeTIMIT' -samples 120 -scale 1.3 -detector dlib -workers 8
if __name__ == '__main__':
    args = parse()
    # main('/share/home/zhangchao/datasets_io03_ssd/DeepfakeTIMIT', 100, 1.3)
    main(args.path, args.samples, args.scale, args.detector, args.workers)
