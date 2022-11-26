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

!!!!!!! important !!!!!!!
jar xvf dfdc_train_part_23.zip
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
    'FAKE': '1',
    'REAL': '0',
}


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    idx=int(sys.argv[1])
    if idx<5:
        print('parsing train data...')
        with open(os.path.join(faces_path, str(idx)+'train.txt'), 'w') as f:
            for i in range(idx*10+3, idx*10+4):
                print(f'dfdc_train_part_{i}')
                rela_path = os.path.join('train', f'dfdc_train_part_{i}')
                with open(os.path.join(path, rela_path, 'metadata.json')) as f_json:
                    metadata = json.load(f_json)
                videos = []
                for video in metadata.keys():
                    videos.append(video)
                for video in tqdm(videos):
                    parse_video(
                        path,
                        rela_path,
                        video,
                        lb[metadata[video]['label']],
                        f,
                        samples,
                        face_scale,
                    )
                print(f'part_{i} done')
        print('parsing train data done')
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
            print('parsing val data done')
    if idx ==6:
        print('parsing test data...')
        with open(os.path.join(faces_path, 'test.txt'), 'w') as f, open(
            os.path.join(path, 'test', 'metadata.json')
        ) as f_json:
            metadata = json.load(f_json)
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
        print('parsing test data done')


def gen_txt(path):
    faces_path = os.path.join(path,'faces')
    train_txt = []
    for i in range(0,50):
        with open(os.path.join(path, 'train', 'dfdc_train_part_'+str(i), 'metadata.json')) as f_json:
            metadata = json.load(f_json)
        for file in os.listdir(os.path.join(faces_path, 'train', 'dfdc_train_part_'+str(i))):
            train_txt.append(os.path.join('faces','train','dfdc_train_part_'+str(i),file)+'\t'+lb[metadata[file[:-6]+'.mp4']['label']]+'\n')

    val_txt = []
    val_dict = {}
    with open(os.path.join(faces_path, 'val.txt'), 'w') as f, open(
            os.path.join(path, 'validation', 'labels.csv'), 'r'
        ) as f_csv:
            csv_reader = csv.reader(f_csv)
            csv_reader.__next__()
            for video, label in tqdm(csv_reader):
                val_dict[video]=label
    for file in os.listdir(os.path.join(faces_path, 'validation')):
        val_txt.append(os.path.join('faces','validation',file)+'\t'+val_dict[file[:-6]+'.mp4']+'\n')

    test_txt = []
    with open(os.path.join(faces_path, 'test.txt'), 'w') as f, open(
            os.path.join(path, 'test', 'metadata.json')
        ) as f_json:
        metadata = json.load(f_json)
    for file in os.listdir(os.path.join(faces_path, 'test')):
        test_txt.append(os.path.join('faces','test',file)+'\t'+str(metadata[file[:-6]+'.mp4']['is_fake'])+'\n')
    print('train:', len(train_txt),train_txt[1000])
    print('val:', len(val_txt),val_txt[1000])
    print('test:', len(test_txt),test_txt[1000])
    with open(os.path.join(faces_path,'train.txt'),'w') as f:
        f.writelines(train_txt)
    with open(os.path.join(faces_path,'val.txt'),'w') as f:
        f.writelines(val_txt)
    with open(os.path.join(faces_path,'test.txt'),'w') as f:
        f.writelines(test_txt)
    all_txt = train_txt+val_txt+test_txt
    with open(os.path.join(faces_path,'all.txt'),'w') as f:
        f.writelines(all_txt)




#  1 is fake
if __name__ == '__main__':
    path = '/share/home/zhangchao/datasets_io03_ssd/DFDC'
    # args = parse()
    # main(args.path, args.samples, args.scale)
    # main('/share/home/zhangchao/datasets_io03_ssd/DFDC',10,1.3)
    gen_txt(path)
