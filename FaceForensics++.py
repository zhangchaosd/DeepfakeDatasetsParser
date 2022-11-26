import sys
import os
import json

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
)


def get_source_videos(videos):
    res = []
    for video in videos:
        res.append(video[:3] + '.mp4')
        res.append(video[4:])
    return res


def double_videos(videos):
    res = []
    for video in videos:
        res.append(video[4:7] + '_' + video[:3] + '.mp4')
    return res + videos


def json2list(js):
    with open(js,'r') as f:
        ls = json.load(f)
    res=[]
    for p in ls:
        res.append(p[0]+'_'+p[1]+'.mp4')
        res.append(p[1]+'_'+p[0]+'.mp4')
        res.append(p[1]+'.mp4')
        res.append(p[0]+'.mp4')
    return res


# https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
def get_ff_splits():
    train_js = './ff++split/train.json'
    val_js = './ff++split/val.json'
    test_js = './ff++split/test.json'
    return json2list(train_js),json2list(val_js),json2list(test_js)


def get_DFD_splits(original_path, manipulated_path):
    train_list = [
        ['01', '02', '03'],
        ['04', '06', '07'],
        ['09', '11', '12', '13'],
        ['14', '15', '18', '20'],
        ['21', '25', '26', '27'],
    ]
    val_list = ['05', '08', '16', '17', '28']
    test_list = ['10', '19', '22', '23', '24']
    idx = int(sys.argv[1])
    train_list = train_list[idx]
    val_list = val_list[idx:idx+1]
    test_list = test_list[idx:idx+1]
    videos = get_files_from_path(original_path) + get_files_from_path(
        manipulated_path
    )
    train_split = []
    val_split = []
    test_split = []
    for video in videos:
        if video[:2] in train_list:
            train_split.append(video)
        elif video[:2] in val_list:
            val_split.append(video)
        elif video[:2] in test_list:
            test_split.append(video)
        # else:
            # assert False
    return train_split, val_split, test_split


def bina_mask(mask):
    mask[mask>0]=255
    return mask


def solve(
    dataset_path,
    faces_path,
    video_path,
    video_name,
    raw_f,
    c23_f,
    c40_f,
    label,
    samples,
    face_scale,
    has_masks,
):
    order, file_names, raw_frames = video2frames(
        os.path.join(dataset_path, video_path, video_name), samples
    )
    if file_names == None:
        return
    _,_, c23_frames = video2frames(
        os.path.join(dataset_path, video_path.replace('raw', 'c23'), video_name), samples, order
    )
    _,_, c40_frames = video2frames(
        os.path.join(dataset_path, video_path.replace('raw', 'c40'), video_name), samples, order
    )
    if has_masks:
        _,_, masks_frames = video2frames(
            os.path.join(dataset_path, video_path.replace('raw', 'masks'), video_name), samples, order
        )
    
    crop_datas = [
        *map(get_face_location, raw_frames, [face_scale] * len(raw_frames))
    ]
    file_names = [
        file_name
        for i, file_name in enumerate(file_names)
        if crop_datas[i] is not None
    ]
    raw_file_names = [
        os.path.join(faces_path, video_path, file_name)
        for file_name in file_names
    ]
    c23_file_names = [
        os.path.join(faces_path, video_path.replace('raw', 'c23'), file_name)
        for file_name in file_names
    ]
    c40_file_names = [
        os.path.join(faces_path, video_path.replace('raw', 'c40'), file_name)
        for file_name in file_names
    ]
    if has_masks:
        masks_file_names = [
            os.path.join(faces_path, video_path.replace('raw', 'masks'), file_name)
            for file_name in file_names
        ]
    raw_frames = [
        frame for i, frame in enumerate(raw_frames) if crop_datas[i] is not None
    ]
    c23_frames = [
        frame for i, frame in enumerate(c23_frames) if crop_datas[i] is not None
    ]
    c40_frames = [
        frame for i, frame in enumerate(c40_frames) if crop_datas[i] is not None
    ]
    if has_masks:
        masks_frames = masks_frames[:len(crop_datas)]
        masks_frames = [
            frame for i, frame in enumerate(masks_frames) if crop_datas[i] is not None
        ]
    crop_datas = [
        crop_data for crop_data in crop_datas if crop_data is not None
    ]
    assert len(raw_frames) == len(c23_frames)
    assert len(raw_frames) == len(c40_frames)
    if has_masks:
        if len(raw_frames) != len(masks_frames):
            print('raw s != masks',len(raw_frames),len(masks_frames))
            return
    assert len(raw_frames) == len(file_names)
    assert len(raw_frames) == len(crop_datas)
    raw_faces = [*map(crop_img, raw_frames, crop_datas)]
    [
        *map(
            cv2.imwrite,
            raw_file_names,
            raw_faces,
            [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(raw_faces),
        )
    ]
    raw_f.writelines(
        [
            os.path.join('faces', video_path, os.path.basename(img_name))
            + '\t'
            + label
            + '' if not has_masks else '\t'+os.path.join('faces', video_path.replace('raw', 'masks'), os.path.basename(img_name))
            + '\n'
            for img_name in file_names
        ]
    )

    c23_faces = [*map(crop_img, c23_frames, crop_datas)]
    [
        *map(
            cv2.imwrite,
            c23_file_names,
            c23_faces,
            [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(c23_faces),
        )
    ]
    c23_f.writelines(
        [
            os.path.join('faces', video_path.replace('raw', 'c23'), os.path.basename(img_name))
            + ' '
            + label
            + '' if not has_masks else ' '+os.path.join('faces', video_path.replace('raw', 'masks'), os.path.basename(img_name))
            + '\n'
            for img_name in file_names
        ]
    )

    c40_faces = [*map(crop_img, c40_frames, crop_datas)]
    [
        *map(
            cv2.imwrite,
            c40_file_names,
            c40_faces,
            [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(c40_faces),
        )
    ]
    c40_f.writelines(
        [
            os.path.join('faces', video_path.replace('raw', 'c40'), os.path.basename(img_name))
            + ' '
            + label
            + '' if not has_masks else ' '+os.path.join('faces', video_path.replace('raw', 'masks'), os.path.basename(img_name))
            + '\n'
            for img_name in file_names
        ]
    )
    if has_masks:
        masks_faces = [*map(crop_img, masks_frames, crop_datas)]
        masks_faces = [*map(bina_mask, masks_faces)]
        [
            *map(
                cv2.imwrite,
                masks_file_names,
                masks_faces,
                [[int(cv2.IMWRITE_JPEG_QUALITY), 100]] * len(masks_faces),
            )
        ]


def main(path, samples, face_scale, subset):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)




    # FaceShifter don't have masks
    if subset == 'FF':
        datasets = [
            'original',
            'Deepfakes',
            'Face2Face',
            'FaceSwap',
            'NeuralTextures',
        ]
        idx = sys.argv[1]
        assert idx in datasets
        datasets = [idx]
        train_split, val_split, test_split = get_ff_splits(
            # os.path.join(
            #     path, 'manipulated_sequences', 'Deepfakes', 'raw', 'videos'
            # )
        )
    else:
        idx = int(sys.argv[1])
        assert idx>=0 and idx<5
        datasets = [
            # 'DeepFakeDetection_original',  # 363
            'DeepFakeDetection',  # 3068
        ]
        train_split, val_split, test_split = get_DFD_splits(
            os.path.join(
                path,
                'original_sequences',
                'actors',
                'raw',
                'videos',
            ),
            os.path.join(
                path,
                'manipulated_sequences',
                'DeepFakeDetection',
                'raw',
                'videos',
            ),
        )
    for i, dataset in enumerate(datasets):
        print(f'Now parsing {dataset}...')
        dataset_i = dataset
        if dataset_i == 'DeepFakeDetection':
            print(f'Now parsing {sys.argv[1]}...')
            dataset_i = dataset_i + '_' + sys.argv[1] +'_'
        f_train_raw = open(os.path.join(faces_path, subset + '_' + dataset_i + '_train_raw.txt'), 'w')
        f_val_raw = open(os.path.join(faces_path, subset + '_' + dataset_i + '_val_raw.txt'), 'w')
        f_test_raw = open(os.path.join(faces_path, subset + '_' + dataset_i + '_test_raw.txt'), 'w')
        f_train_c23 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_train_c23.txt'), 'w')
        f_val_c23 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_val_c23.txt'), 'w')
        f_test_c23 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_test_c23.txt'), 'w')
        f_train_c40 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_train_c40.txt'), 'w')
        f_val_c40 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_val_c40.txt'), 'w')
        f_test_c40 = open(os.path.join(faces_path, subset + '_' + dataset_i + '_test_c40.txt'), 'w')
        label = '0 '
        if 'original' == dataset:
            raw_path = os.path.join(
                'original_sequences', 'youtube', 'raw', 'videos'
            )
        elif 'DeepFakeDetection_original' == dataset:
            raw_path = os.path.join(
                'original_sequences', 'actors', 'raw', 'videos'
            )
        else:
            raw_path = 'manipulated_sequences'
            raw_path = os.path.join(raw_path, dataset, 'raw', 'videos')
            label = '1 '
        label = label + str(i)

        gen_dirs(os.path.join(faces_path, raw_path))
        c23_path = raw_path.replace('raw', 'c23')
        gen_dirs(os.path.join(faces_path, c23_path))
        c40_path = raw_path.replace('raw', 'c40')
        gen_dirs(os.path.join(faces_path, c40_path))
        if 'original' not in dataset:
            masks_path = raw_path.replace('raw', 'masks')
            gen_dirs(os.path.join(faces_path, masks_path))

        raw_videos = get_files_from_path(os.path.join(path, raw_path))
        raw_videos = raw_videos[2978:]
        for video in tqdm(raw_videos):
            if video in train_split:
                f_raw = f_train_raw
                f_c23 = f_train_c23
                f_c40 = f_train_c40
            elif video in val_split:
                f_raw = f_val_raw
                f_c23 = f_val_c23
                f_c40 = f_val_c40
            elif video in test_split:
                f_raw = f_test_raw
                f_c23 = f_test_c23
                f_c40 = f_test_c40
            else:
                continue

            solve(
                path,
                faces_path,
                raw_path,
                video,
                f_raw,
                f_c23,
                f_c40,
                label,
                samples,
                face_scale,
                'original' not in dataset,
            )
        f_train_raw.close()
        f_val_raw.close()
        f_test_raw.close()
        f_train_c23.close()
        f_val_c23.close()
        f_test_c23.close()
        f_train_c40.close()
        f_val_c40.close()
        f_test_c40.close()


'''
change code 'return' to 'continue' to download the full datasets in download_ffdf.py
'''
'''
usage:
  FF:
    python FaceForensics++.py original
    python FaceForensics++.py Deepfakes
    python FaceForensics++.py Face2Face
    python FaceForensics++.py FaceSwap
    python FaceForensics++.py NeuralTextures
    regen_ff_txt()
  DFD:
    python FaceForensics++.py 0
    python FaceForensics++.py 1
    python FaceForensics++.py 2
    python FaceForensics++.py 3
    python FaceForensics++.py 4
'''
def list2set(split):
    split = [s[:3] for s in split]
    return set(split)
def check_exist(txt,path):
    for line in txt:
        ss = line.strip().split()
        pa = os.path.join(path,ss[0])
        if not os.path.exists(pa):
            print('!!!!!',pa)
        if len(ss)==3:
            pa = os.path.join(path,ss[2])
            if not os.path.exists(pa):
                print('!!!!!',pa)
# txt files are wrong, so call this function to regenerate these txt files
def regen_ff_txt(path):
    subsets = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    comps = ['raw','c23','c40']
    modes = ['train','val','test']
    train_split, val_split, test_split = get_ff_splits()
    train_split = list2set(train_split)
    val_split = list2set(val_split)
    test_split = list2set(test_split)
    # raw
    train_txt = []
    val_txt = []
    test_txt = []
    files = os.listdir(os.path.join(path,'faces','original_sequences','youtube','raw','videos'))
    for file in files:
        info = os.path.join('faces','original_sequences','youtube','raw','videos',file) + '\t0\n'
        if file[:3] in train_split:
            train_txt.append(info)
        elif file[:3] in val_split:
            val_txt.append(info)
        elif file[:3] in test_split:
            test_txt.append(info)
        else:
            assert False
    for subset in subsets:
        files = os.listdir(os.path.join(path,'faces','manipulated_sequences',subset,'raw','videos'))
        for file in files:
            info = os.path.join('faces','manipulated_sequences',subset,'raw','videos',file)
            info = info + '\t1\t' + info.replace('raw', 'masks') + '\n'
            if file[:3] in train_split:
                train_txt.append(info)
            elif file[:3] in val_split:
                val_txt.append(info)
            elif file[:3] in test_split:
                test_txt.append(info)
            else:
                assert False
    print(len(train_txt))
    print(len(val_txt))
    print(len(test_txt))
    check_exist(train_txt,path)
    check_exist(val_txt,path)
    check_exist(test_txt,path)
    with open(os.path.join(path,'faces','ff_train_raw.txt'),'w') as f:
        f.writelines(train_txt)
    with open(os.path.join(path,'faces','ff_val_raw.txt'),'w') as f:
        f.writelines(val_txt)
    with open(os.path.join(path,'faces','ff_test_raw.txt'),'w') as f:
        f.writelines(test_txt)
    train_txt = [l.replace('raw','c23') for l in train_txt]
    val_txt = [l.replace('raw','c23') for l in val_txt]
    test_txt = [l.replace('raw','c23') for l in test_txt]
    check_exist(train_txt,path)
    check_exist(val_txt,path)
    check_exist(test_txt,path)
    with open(os.path.join(path,'faces','ff_train_c23.txt'),'w') as f:
        f.writelines(train_txt)
    with open(os.path.join(path,'faces','ff_val_c23.txt'),'w') as f:
        f.writelines(val_txt)
    with open(os.path.join(path,'faces','ff_test_c23.txt'),'w') as f:
        f.writelines(test_txt)
    train_txt = [l.replace('c23','c40') for l in train_txt]
    val_txt = [l.replace('c23','c40') for l in val_txt]
    test_txt = [l.replace('c23','c40') for l in test_txt]
    check_exist(train_txt,path)
    check_exist(val_txt,path)
    check_exist(test_txt,path)
    with open(os.path.join(path,'faces','ff_train_c40.txt'),'w') as f:
        f.writelines(train_txt)
    with open(os.path.join(path,'faces','ff_val_c40.txt'),'w') as f:
        f.writelines(val_txt)
    with open(os.path.join(path,'faces','ff_test_c40.txt'),'w') as f:
        f.writelines(test_txt)

def regen_dfd_txt(path):
    train_split = [
        '01', '02', '03',
        '04', '06', '07',
        '09', '11', '12', '13',
        '14', '15', '18', '20',
        '21', '25', '26', '27',
    ]
    val_split = ['05', '08', '16', '17', '28']
    test_split = ['10', '19', '22', '23', '24']
    # raw
    train_txt = []
    val_txt = []
    test_txt = []
    raw_path = os.path.join(path,'faces','original_sequences','actors','raw','videos')
    c23_path = raw_path.replace('raw', 'c23')
    c40_path = raw_path.replace('raw', 'c40')
    masks_path = raw_path.replace('raw', 'masks')
    raw_files = os.listdir(raw_path)
    for file in raw_files:
        if not (os.path.exists(os.path.join(c23_path,file)) and os.path.exists(os.path.join(c40_path,file))):
            print(file, 'not complete')
            continue
        info = os.path.join('faces','original_sequences','actors','raw','videos',file) + '\t0\n'
        if file[:2] in train_split:
            train_txt.append(info)
        elif file[:2] in val_split:
            val_txt.append(info)
        elif file[:2] in test_split:
            test_txt.append(info)
        else:
            assert False
    raw_path = os.path.join(path,'faces','manipulated_sequences','DeepFakeDetection','raw','videos')
    c23_path = raw_path.replace('raw', 'c23')
    c40_path = raw_path.replace('raw', 'c40')
    masks_path = raw_path.replace('raw', 'masks')
    raw_files = os.listdir(raw_path)
    for file in raw_files:
        if not (os.path.exists(os.path.join(c23_path,file)) and os.path.exists(os.path.join(c40_path,file)) and os.path.exists(os.path.join(masks_path,file))):
            print(file, 'not complete 2')
            continue
        info = os.path.join('faces','manipulated_sequences','DeepFakeDetection','raw','videos',file)
        info = info + '\t1\t' + info.replace('raw', 'masks') + '\n'
        if file[:2] in train_split:
            train_txt.append(info)
        elif file[:2] in val_split:
            val_txt.append(info)
        elif file[:2] in test_split:
            test_txt.append(info)
        else:
            assert False
    print(len(train_txt))
    print(len(val_txt))
    print(len(test_txt))
    # check_exist(train_txt,path)
    # check_exist(val_txt,path)
    # check_exist(test_txt,path)
    with open(os.path.join(path,'faces','dfd_train_raw.txt'),'w') as f:
        f.writelines(train_txt)
    with open(os.path.join(path,'faces','dfd_val_raw.txt'),'w') as f:
        f.writelines(val_txt)
    with open(os.path.join(path,'faces','dfd_test_raw.txt'),'w') as f:
        f.writelines(test_txt)
    c23_train_txt = [l.replace('raw','c23') for l in train_txt]
    c23_val_txt = [l.replace('raw','c23') for l in val_txt]
    c23_test_txt = [l.replace('raw','c23') for l in test_txt]
    # check_exist(train_txt,path)
    # check_exist(val_txt,path)
    # check_exist(test_txt,path)
    with open(os.path.join(path,'faces','dfd_train_c23.txt'),'w') as f:
        f.writelines(c23_train_txt)
    with open(os.path.join(path,'faces','dfd_val_c23.txt'),'w') as f:
        f.writelines(c23_val_txt)
    with open(os.path.join(path,'faces','dfd_test_c23.txt'),'w') as f:
        f.writelines(c23_test_txt)
    c40_train_txt = [l.replace('raw','c40') for l in train_txt]
    c40_val_txt = [l.replace('raw','c40') for l in val_txt]
    c40_test_txt = [l.replace('raw','c40') for l in test_txt]
    # check_exist(train_txt,path)
    # check_exist(val_txt,path)
    # check_exist(test_txt,path)
    with open(os.path.join(path,'faces','dfd_train_c40.txt'),'w') as f:
        f.writelines(c40_train_txt)
    with open(os.path.join(path,'faces','dfd_val_c40.txt'),'w') as f:
        f.writelines(c40_val_txt)
    with open(os.path.join(path,'faces','dfd_test_c40.txt'),'w') as f:
        f.writelines(c40_test_txt)
    raw_all_txt = train_txt +val_txt+ test_txt
    c23_all_txt = c23_train_txt + c23_val_txt + c23_test_txt
    c40_all_txt = c40_train_txt + c40_val_txt + c40_test_txt
    with open(os.path.join(path,'faces','dfd_all_raw.txt'),'w') as f:
        f.writelines(raw_all_txt)
    with open(os.path.join(path,'faces','dfd_all_c23.txt'),'w') as f:
        f.writelines(c23_all_txt)
    with open(os.path.join(path,'faces','dfd_all_c40.txt'),'w') as f:
        f.writelines(c40_all_txt)

if __name__ == '__main__':
    dataset_path = '/share/home/zhangchao/datasets_io03_ssd/ff++'
    # main('/share/home/zhangchao/datasets_io03_ssd/ff++', 50, 1.3, 'FF')
    # main('/share/home/zhangchao/datasets_io03_ssd/ff++', 50, 1.3, 'DFD')
    regen_dfd_txt(dataset_path)

    exit()
    args = parse()
    main(args.path, args.samples, args.scale, args.subset)
