import sys
import csv
import json
import os

from tqdm import tqdm

from utils import gen_dirs, parse, parse_video

'''
#!/usr/bin/env bash

for i in {0..49}
do
    unzip dfdc_train_part_${i}.zip
done
'''
'''
python DFDC.py 0
python DFDC.py 1
python DFDC.py 2
python DFDC.py 3
python DFDC.py 4
python DFDC.py 5
python DFDC.py 6
'''

lb = {
    'FAKE': 1,
    'REAL': 0,
}


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    idx=int(sys.argv[1])
    print('parsing train data...')
    with open(os.path.join(faces_path, 'train.txt'), 'w') as f:
        for i in range(idx*10, idx*10+10):
            print(f'dfdc_train_part_{i}')
            rela_path = os.path.join('train', f'dfdc_train_part_{i}')
            with open(os.path.join(path, rela_path, 'metadata.json')) as f_json:
                metadata = json.load(f_json)
                for video in tqdm(metadata.keys()):
                    parse_video(
                        path,
                        rela_path,
                        video,
                        lb[metadata[video]['label']],
                        f,
                        samples,
                        face_scale,
                    )
    if idx == 5:
        print('parsing val data...')
        with open(os.path.join(faces_path, 'val.txt'), 'w') as f, open(
            os.path.join(path, 'validation', 'labels.csv'), 'r'
        ) as f_csv:
            csv_reader = csv.reader(f_csv)
            csv_reader.__next__()
            for video, label in tqdm(csv_reader):
                parse_video(
                    path, 'validation', video, label, f, samples, face_scale
                )
    if idx ==6:
        print('parsing test data...')
        with open(os.path.join(faces_path, 'test.txt'), 'w') as f, open(
            os.path.join(path, rela_path, 'metadata.json')
        ) as f_json:
            for video in tqdm(metadata.keys()):
                parse_video(
                    path,
                    'test',
                    video,
                    metadata[video]['is_fake'],
                    f,
                    samples,
                    face_scale,
                )


#  1 is fake
if __name__ == '__main__':
    # args = parse()
    # main(args.path, args.samples, args.scale)
    main('/share/home/zhangchao/datasets_io03_ssd/DFDC',10,1.3)
