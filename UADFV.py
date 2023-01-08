import multiprocessing as mp
import os
from functools import partial

from tqdm import tqdm

from utils import (
    gen_dirs,
    get_files_from_path,
    parse,
    static_shuffle,
    video2face_pngs,
)


def get_all_videos(real_videos):
    fake_videos = []
    for v in real_videos:
        fake_videos.append(v[:-4].replace('real', 'fake') + '_fake.mp4')
    return fake_videos + real_videos


def get_splits(videos):
    static_shuffle(videos)
    return (
        get_all_videos(videos[:30]),
        get_all_videos(videos[30:39]),
        get_all_videos(videos[39:]),
    )  # 30*2, 9*2, 10*2


def f1(video_path, faces_path, samples, face_scale, detector):
    if video_path[-9] == '_':
        k = 'fake'
        label = '1'
    else:
        k = 'real'
        label = '0'
    save_path = os.path.join(faces_path, k)
    infos = [
        os.path.join(save_path, img_name) + '\t' + label + '\n'
        for img_name in video2face_pngs(
            video_path, save_path, samples, face_scale, detector
        )
    ]
    return infos


def f2(mode, split, faces_path, samples, face_scale, detector, num_workers):
    txt_path = os.path.join(faces_path, mode + '.txt')
    infos = []
    with mp.Pool(num_workers) as workers, open(txt_path, 'w') as f:
        with tqdm(total=len(split)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    f1,
                    faces_path=faces_path,
                    samples=samples,
                    face_scale=face_scale,
                    detector=detector,
                ),
                split,
            ):
                pbar.update()
                infos += info
        f.writelines(infos)
    return infos


def main(path, samples, face_scale, detector, num_workers):
    faces_path = os.path.join(path, 'faces_test')
    gen_dirs(faces_path)
    fake_path = os.path.join(path, 'fake')
    real_path = os.path.join(path, 'real')
    fake_videos = get_files_from_path(fake_path)
    real_videos = get_files_from_path(real_path)
    assert len(fake_videos) == 49
    assert len(real_videos) == 49
    real_videos = [os.path.join(real_path, video) for video in real_videos]
    train_split, val_split, test_split = get_splits(real_videos)

    train_info = f2(
        'train',
        train_split,
        faces_path,
        samples,
        face_scale,
        detector,
        num_workers,
    )
    val_info = f2(
        'val',
        val_split,
        faces_path,
        samples,
        face_scale,
        detector,
        num_workers,
    )
    test_info = f2(
        'test',
        test_split,
        faces_path,
        samples,
        face_scale,
        detector,
        num_workers,
    )

    all_info = train_info + val_info + test_info
    all_txt = os.path.join(faces_path, 'all.txt')
    with open(all_txt, 'w') as f:
        f.writelines(all_info)
    print('Num of train: ', len(train_info))
    print('Num of val: ', len(val_info))
    print('Num of test: ', len(test_info))
    print('Total: ', len(all_info))


# 0 is real
# python UADFV.py -path '/share/home/zhangchao/datasets_io03_ssd/UADFV' -samples 100 -scale 1.3 -detector dlib -workers 8
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale, args.detector, args.workers)
