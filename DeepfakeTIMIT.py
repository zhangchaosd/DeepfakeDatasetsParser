import multiprocessing as mp
import os
from functools import partial

import cv2
from tqdm import tqdm

from utils import (
    gen_dirs,
    get_files_from_path,
    parse,
    static_shuffle,
    video2face_pngs,
    img2face,
)



def parse_video(video_path, path, path_auth, save_path, faces_prefix, samples, face_scale, detector):
    rela_path = os.path.dirname(video_path)
    save_path_fake = os.path.join(save_path, faces_prefix, rela_path)
    video_path = os.path.join(path, video_path)
    infos = [
        os.path.join(rela_path, img_name) + '\t1\n'
        for img_name in video2face_pngs(
            video_path, save_path_fake, samples, face_scale, detector
        )
    ]
    subject = rela_path.split('/')[1]
    pos = os.path.basename(video_path).split('-')[0]
    rela_path = os.path.join(subject, 'video', pos)
    gen_dirs(os.path.join(save_path, faces_prefix, rela_path))
    jpegs = get_files_from_path(os.path.join(path_auth, rela_path))
    imgs = [cv2.imread(os.path.join(path_auth,rela_path, jpeg)) for jpeg in jpegs]
    faces = map(partial(img2face,face_scale=face_scale,detector=detector), imgs)
    img_paths = [os.path.join(rela_path, jpeg+'.png') for jpeg in jpegs]
    list(map(cv2.imwrite, [os.path.join(save_path, faces_prefix, img_path) for img_path in img_paths], faces))
    infos += [img_path+'\t0\n' for img_path in img_paths]
    return infos

def parse_split(mode, split, path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers):
    videos = []
    for subject in split:
        videos += [os.path.join(subject, file) for file in get_files_from_path(os.path.join(path, subject)) if file.endswith('.avi')]

    txt_path = os.path.join(save_path, faces_prefix, mode + '.txt')
    infos = []

    with mp.Pool(num_workers) as workers, open(txt_path, 'w') as f:
        with tqdm(total=len(videos)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    parse_video,
                    path=path,
                    path_auth=path_auth,
                    save_path=save_path,
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

def parse_quality(quality, path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers):
    gen_dirs(os.path.join(save_path, faces_prefix, quality))
    # subjects = get_files_from_path(os.path.join(path, faces_prefix, quality))
    subjects = [os.path.join(quality, subject) for subject in get_files_from_path(os.path.join(path, quality))]
    assert len(subjects) == 32
    static_shuffle(subjects)
    train_split = subjects[:22]
    test_split = subjects[22:]
    all_info = []
    print('Parsing train split...')
    all_info += parse_split('train_'+quality, train_split, path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Parsing test split...')
    all_info += parse_split('test_'+quality, test_split, path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers)
    with open(os.path.join(save_path, faces_prefix, 'all_'+quality+'.txt'), 'w') as f:
        f.writelines(all_info)
    print('All ', len(all_info))

def main(path, path_auth, save_path, samples, face_scale, detector, num_workers):
    faces_prefix = 'faces' + str(samples) + detector
    gen_dirs(os.path.join(save_path, faces_prefix))
    print('Parsing lower_quality...')
    parse_quality('lower_quality', path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Parsing higher_quality...')
    parse_quality('higher_quality', path, path_auth, save_path, faces_prefix, samples, face_scale, detector, num_workers)
    print('Done')


# 1 is fake
# python DeepfakeTIMIT.py -path '/share/home/zhangchao/datasets_io03_ssd/DeepfakeTIMIT' -path_auth '/share/home/zhangchao/datasets_io03_ssd/vidtimit' -save_path '/share/home/zhangchao/local_sets/DeepfakeTIMITdebug' -samples 20 -scale 1.3 -detector dlib -workers 12
if __name__ == '__main__':
    args = parse()
    main(args.path, args.path_auth, args.save_path, args.samples, args.scale, args.detector, args.workers)
