import multiprocessing as mp
import os
from functools import partial
import json

import cv2
from tqdm import tqdm
from utils import (
    crop_img,
    gen_dirs,
    get_face_location,
    get_files_from_path,
    parse,
    video2frames,
)


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
    return json2list(train_js), json2list(val_js), json2list(test_js)


def get_DFD_splits(path):
    train_list = [
        ['01', '02', '03'],
        ['04', '06', '07'],
        ['09', '11', '12', '13'],
        ['14', '15', '18', '20'],
        ['21', '25', '26', '27'],
    ]
    val_list = ['05', '08', '16', '17', '28']
    test_list = ['10', '19', '22', '23', '24']

    videos = get_files_from_path(
        os.path.join(
            path,
            'original_sequences',
            'actors',
            'raw',
            'videos',
        )) + get_files_from_path(
        os.path.join(
            path,
            'manipulated_sequences',
            'DeepFakeDetection',
            'raw',
            'videos',
        )
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
        else:
            assert False
    return train_split, val_split, test_split


def bina_mask(mask):
    mask[mask>0]=255
    return mask


def f4(frames, crop_datas, save_path):
    [
        *map(
            cv2.imwrite,
            [os.path.join(save_path, fn) for _,fn,_ in frames],
            map(crop_img, [frame for _,_,frame in frames], crop_datas),
        )
    ]
    pass


def f3(video, label, path, rela_path, faces_prefix, samples, face_scale, detector):
    raw_rela_path = os.path.join(rela_path,'raw','videos')
    c23_rela_path = os.path.join(rela_path,'c23','videos')
    c40_rela_path = os.path.join(rela_path,'c40','videos')
    gen_dirs(os.path.join(path, faces_prefix, raw_rela_path))
    gen_dirs(os.path.join(path, faces_prefix, c23_rela_path))
    gen_dirs(os.path.join(path, faces_prefix, c40_rela_path))
    raw_frames = list(video2frames(os.path.join(path, raw_rela_path, video), samples))  # orders, file_names, raw_frames
    crop_datas = [*map(partial(get_face_location, face_scale=face_scale, detector=detector), raw_frames[2])]
    # raw_frames = [(raw_frames[0][i],raw_frames[1][i],raw_frames[2][i]) for i in range(len(raw_frames[0])) if crop_datas[i] is not None]
    raw_frames[0] = [raw_frames[0][i] for i in range(len(raw_frames[0])) if crop_datas[i] is not None]
    raw_frames[1] = [raw_frames[1][i] for i in range(len(raw_frames[1])) if crop_datas[i] is not None]
    raw_frames[2] = [raw_frames[2][i] for i in range(len(raw_frames[2])) if crop_datas[i] is not None]
    c23_frames = video2frames(os.path.join(path,c23_rela_path,video), samples)
    c40_frames = video2frames(os.path.join(path,c40_rela_path,video), samples)

    if label == '1':
        masks_rela_path = os.path.join(path,rela_path,'masks','videos',video)
        gen_dirs(os.path.join(path, masks_rela_path))
        masks_frames = list(video2frames(os.path.join(path,masks_rela_path,video), samples))
        final_orders = list(set(raw_frames[0]) & set(c23_frames[0]) & set(c40_frames[0]) & set(masks_frames[0]))
    else:
        final_orders = list(set(raw_frames[0]) & set(c23_frames[0]) & set(c40_frames[0]))
    crop_datas = [crop_data for i, crop_data in enumerate(crop_datas) if raw_frames[0][i] in final_orders]
    raw_frames = [(frame_ind, raw_frames[1][i], raw_frames[2][i]) for i, frame_ind in enumerate(raw_frames[0]) if frame_ind in final_orders]
    c23_frames = [(frame_ind, c23_frames[1][i], c23_frames[2][i]) for i, frame_ind in enumerate(c23_frames[0]) if frame_ind in final_orders]
    c40_frames = [(frame_ind, c40_frames[1][i], c40_frames[2][i]) for i, frame_ind in enumerate(c40_frames[0]) if frame_ind in final_orders]
    f4(raw_frames, crop_datas, os.path.join(path,faces_prefix,raw_rela_path))
    f4(c23_frames, crop_datas, os.path.join(path,faces_prefix,c23_rela_path))
    f4(c40_frames, crop_datas, os.path.join(path,faces_prefix,c40_rela_path))
    raw_infos = [os.path.join(faces_prefix,raw_rela_path,fn)+f'\t{label}\n' for _,fn,_ in raw_frames]
    c23_infos = [os.path.join(faces_prefix,c23_rela_path,fn)+f'\t{label}\n' for _,fn,_ in c23_frames]
    c40_infos = [os.path.join(faces_prefix,c40_rela_path,fn)+f'\t{label}\n' for _,fn,_ in c40_frames]
    if label =='1':  # FaceShifter will ERR
        masks_frames = [(frame_ind, masks_frames[1][i], masks_frames[2][i]) for i, frame_ind in enumerate(masks_frames[0]) if frame_ind in final_orders]
        masks_frames[2] = [*map(bina_mask,masks_frames[2])]
        f4(masks_frames, crop_datas, os.path.join(path,faces_prefix,masks_rela_path))
        raw_infos = [info[:-1]+'\t'+os.path.join(faces_prefix,masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(raw_infos)]
        c23_infos = [info[:-1]+'\t'+os.path.join(faces_prefix,masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(c23_infos)]
        c40_infos = [info[:-1]+'\t'+os.path.join(faces_prefix,masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(c40_infos)]
    return raw_infos, c23_infos, c40_infos

def f5(txt_path, infos):
    with open(txt_path, 'w') as f:
        f.writelines(infos)

def f1(mode, split, datasets, subset, faces_prefix, path, samples, face_scale, detector, num_workers):
    print(f'Now parsing {subset} {mode}...')
    raw_infos = []
    c23_infos = []
    c40_infos = []
    for dataset in datasets:
        print(f'Now parsing {dataset}...')
        label = '0'
        if 'original' == dataset:
            rela_path = os.path.join(
                'original_sequences', 'youtube'
            )
        elif 'DeepFakeDetection_original' == dataset:
            rela_path = os.path.join(
                'original_sequences', 'actors'
            )
        else:
            rela_path = os.path.join('manipulated_sequences', dataset)
            label = '1'
        with mp.Pool(num_workers) as workers:
            with tqdm(total=len(split)) as pbar:
                for raw_info, c23_info, c40_info in workers.imap_unordered(
                    partial(
                        f3,
                        label=label,
                        path=path,
                        rela_path=rela_path,
                        faces_prefix=faces_prefix,
                        samples=samples,
                        face_scale=face_scale,
                        detector=detector,
                    ),
                    split,
                ):
                    pbar.update()
                    raw_infos += raw_info
                    c23_infos += c23_info
                    c40_infos += c40_info
    assert len(raw_infos)==len(c23_infos)
    assert len(raw_infos)==len(c40_infos)
    print(mode, datasets, len(raw_infos))
    f5(os.path.join(path, faces_prefix, f'{subset}_{mode}_raw.txt'), raw_infos)
    f5(os.path.join(path, faces_prefix, f'{subset}_{mode}_c23.txt'), c23_infos)
    f5(os.path.join(path, faces_prefix, f'{subset}_{mode}_c40.txt'), c40_infos)
    return raw_infos, c23_infos, c40_infos


def main(subset, path, samples, face_scale, detector, num_workers):
    faces_prefix = f'faces{samples}detector'
    gen_dirs(os.path.join(path, faces_prefix))

    if subset == 'FF':
        # FaceShifter don't have masks
        datasets = [
            'original',
            'Deepfakes',
            'Face2Face',
            'FaceSwap',
            'NeuralTextures',
            # 'FaceShifter',
        ]
        train_split, val_split, test_split = get_ff_splits()
    else:
        datasets = [
            'DeepFakeDetection_original',  # 363
            'DeepFakeDetection',  # 3068
        ]
        train_split, val_split, test_split = get_DFD_splits(path)
    
    train_raw_infos, train_c23_infos, train_c40_infos = f1('train', train_split, datasets, subset, faces_prefix, path, samples, face_scale, detector, num_workers)
    val_raw_infos, val_c23_infos, val_c40_infos = f1('val', val_split, datasets, subset, faces_prefix, path, samples, face_scale, detector, num_workers)
    test_raw_infos, test_c23_infos, test_c40_infos = f1('test', test_split, datasets, subset, faces_prefix, path, samples, face_scale, detector, num_workers)

    f5(os.path.join(path,faces_prefix,f'{subset}_all_raw.txt'),train_raw_infos+val_raw_infos+test_raw_infos)
    f5(os.path.join(path,faces_prefix,f'{subset}_all_c23.txt'),train_c23_infos+val_c23_infos+test_c23_infos)
    f5(os.path.join(path,faces_prefix,f'{subset}_all_c40.txt'),train_c40_infos+val_c40_infos+test_c40_infos)


'''
change code 'return' to 'continue' to download the full datasets in download_ffdf.py
'''
# real is 0
# python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -subset FF -samples 120 -scale 1.3 -detector dlib -workers 8
# python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -subset DFD -samples 120 -scale 1.3 -detector dlib -workers 8
if __name__ == '__main__':
    args = parse()
    main(args.subset, args.path, args.samples, args.scale, args.detector, args.workers)
