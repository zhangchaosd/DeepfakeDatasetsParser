import multiprocessing as mp
import os
from functools import partial

from tqdm import tqdm

from utils import get_files_from_path, parse, gen_dirs, video2face_pngs


def f2(line, path, faces_prefix, samples, face_scale, detector):
    label, video_path = line.split(' ')
    rela_path = os.path.dirname(video_path)
    save_path = os.path.join(path, faces_prefix, rela_path)
    video_path = os.path.join(path, video_path)
    infos = [
        os.path.join(faces_prefix, rela_path, img_name) + f'\t{label}\n'
        for img_name in video2face_pngs(
            video_path, save_path, samples, face_scale, detector
        )
    ]
    return infos

def f1(mode, lines, path, faces_prefix, samples, face_scale, detector, num_workers):
    print(f'Parsing {mode}...')
    txt_path = os.path.join(path, faces_prefix, mode + '.txt')
    infos = []
    with mp.Pool(num_workers) as workers, open(txt_path, 'w') as f:
        with tqdm(total=len(lines)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    f2,
                    path=path,
                    faces_prefix=faces_prefix,
                    samples=samples,
                    face_scale=face_scale,
                    detector=detector,
                ),
                lines,
            ):
                pbar.update()
                infos += info
        f.writelines(infos)
    print(mode, len(infos))
    return infos

def main(path, samples, face_scale, detector, num_workers):
    faces_prefix = 'faces' + str(samples) + detector
    gen_dirs(os.path.join(path, faces_prefix))

    test_txt = os.path.join(path, 'List_of_testing_videos.txt')
    with open(test_txt, 'r') as f:
        test_lines = f.read().splitlines()
    test_videos = []

    for l in test_lines:
        _, video = l.split(' ')
        test_videos.append(video)
    assert len(test_videos) == 100 or len(test_videos) == 518

    real_videos = [
        os.path.join('Celeb-real', video)
        for video in get_files_from_path(os.path.join(path, 'Celeb-real'))
    ]
    real_videos += [
        os.path.join('YouTube-real', video)
        for video in get_files_from_path(os.path.join(path, 'YouTube-real'))
    ]
    synthesis_videos = [
        os.path.join('Celeb-synthesis', video)
        for video in get_files_from_path(os.path.join(path, 'Celeb-synthesis'))
    ]

    real_videos = [
        *filter(lambda v: v not in test_videos, real_videos)
    ]
    synthesis_videos = [
        *filter(lambda v: v not in test_videos, synthesis_videos)
    ]

    train_lines = [('0 ' + v) for v in real_videos] + [
        ('1 ' + v) for v in synthesis_videos
    ]
    assert len(train_lines) == 1103 or len(train_lines) == 6011

    all_info = f1('train', train_lines, path, faces_prefix, samples, face_scale, detector, num_workers)
    all_info += f1('test', test_lines, path, faces_prefix, samples, face_scale, detector, num_workers)
    with open(os.path.join(path, faces_prefix, 'all.txt'), 'w') as f:
        f.writelines(all_info)
    print('All ', len(all_info))



# real is 0
# python CelebDF.py -path '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF' -samples 120 -scale 1.3 -detector dlib -workers 8
# python CelebDF.py -path '/share/home/zhangchao/datasets_io03_ssd/Celeb-DF-v2' -samples 120 -scale 1.3 -detector dlib -workers 8
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale, args.detector, args.workers)
    # print('v1')
    # main('/share/home/zhangchao/datasets_io03_ssd/Celeb-DF', 100, 1.3)
    # print('v2')
    # main('/share/home/zhangchao/datasets_io03_ssd/Celeb-DF-v2', 100, 1.3)
    # print('done')
