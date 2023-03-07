import csv
import json
from functools import partial
import os
import multiprocessing as mp

from tqdm import tqdm

from utils import gen_dirs, parse, video2face_pngs

'''
#!/usr/bin/env bash

for i in {0..49}
do
    unzip dfdc_train_part_${i}.zip
done

!!!!!!! important !!!!!!!
jar xvf dfdc_train_part_23.zip
'''


lb = {
    'FAKE': '1',
    'REAL': '0',
}


def parse_video(video, path, save_path, faces_prefix, samples, face_scale, detector):
    video_path, label = video
    infos = [
        os.path.join(os.path.dirname(video_path), img_name) + '\t' + str(label) + '\n'
        for img_name in video2face_pngs(
            os.path.join(path, video_path), os.path.join(save_path,faces_prefix,os.path.dirname(video_path)), samples, face_scale, detector
        )
    ]
    return infos


def get_videos(path, rela_path):
    videos = []
    if rela_path.startswith('val'):
        with open(os.path.join(path, rela_path, 'labels.csv'), 'r') as f_csv:
            csv_reader = csv.reader(f_csv)
            csv_reader.__next__()
            for video, label in tqdm(csv_reader):
                videos.append((os.path.join(rela_path,video),label))
    else:
        with open(os.path.join(path, rela_path, 'metadata.json')) as f_json:
            metadata = json.load(f_json)
            for video in metadata.keys():
                if rela_path.startswith('tr'):
                    videos.append((os.path.join(rela_path,video),lb[metadata[video]['label']]))
                else:
                    videos.append((os.path.join(rela_path,video),metadata[video]['is_fake']))
    return videos

def main(path, save_path, samples, face_scale, detector, num_workers, mode, part=0):
    assert part>=0 and part<5
    faces_prefix = f'faces{samples}{detector}'
    gen_dirs(os.path.join(save_path, faces_prefix))

    # modes = ['test']
    # modes = ['train', 'validation', 'test']
    modes=[mode]
    all_infos = []
    for mode in modes:
        print(f'Parsing {mode}...')
        videos = []
        if mode == 'train':
            for i in range(part*10,part*10+10):
                videos += get_videos(path, os.path.join(mode, f'dfdc_train_part_{i}'))
        else:
            videos = get_videos(path, mode)
        infos = []
        with mp.Pool(num_workers) as workers:
            with tqdm(total=len(videos)) as pbar:
                for info in workers.imap_unordered(
                    partial(
                        parse_video,
                        path=path,
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
        print(f'Parsing {mode} {len(infos)}')
        with open(os.path.join(save_path, faces_prefix, f'{mode}_{part}.txt'), 'w') as f:
            f.writelines(infos)
        all_infos += infos
    # with open(os.path.join(save_path, faces_prefix, 'all_{part}.txt'), 'w') as f:
        # f.writelines(all_infos)
    print('All done ', len(all_infos))

def merge_txts(save_path,samples,detector):
    faces_prefix = f'faces{samples}{detector}'
    with open(os.path.join(save_path, faces_prefix, 'train.txt'), 'w') as f:
        for i in range(0,5):
            f.writelines(open(os.path.join(save_path, faces_prefix, f'train_{i}.txt')))
    with open(os.path.join(save_path, faces_prefix, 'test.txt'), 'w') as f:
        f.writelines(open(os.path.join(save_path, faces_prefix, 'test_0.txt')))
    with open(os.path.join(save_path, faces_prefix, 'validation.txt'), 'w') as f:
        f.writelines(open(os.path.join(save_path, faces_prefix, 'validation_0.txt')))
    with open(os.path.join(save_path, faces_prefix, 'all.txt'), 'w') as f:
        f.writelines(open(os.path.join(save_path, faces_prefix, 'train.txt')))
        f.writelines(open(os.path.join(save_path, faces_prefix, 'validation.txt')))
        f.writelines(open(os.path.join(save_path, faces_prefix, 'test.txt')))

#  1 is fake
# python DFDC.py -path '/share/home/zhangchao/datasets_io03_ssd/DFDC' -save_path '/share/home/zhangchao/datasets_io03_ssd/DFDC' -samples 20 -scale 1.3 -detector dlib -workers 24 -mode train -part 0
if __name__ == '__main__':
    args = parse()
    # main(args.path, args.save_path, args.samples, args.scale, args.detector, args.workers, args.mode, args.part)
    merge_txts(args.save_path, args.samples, args.detector)
