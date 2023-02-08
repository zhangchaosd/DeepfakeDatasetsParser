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
        '01', '02', '03',
        '04', '06', '07',
        '09', '11', '12', '13',
        '14', '15', '18', '20',
        '21', '25', '26', '27',
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
            print(video)
            assert False
    return train_split, val_split, test_split


def bina_mask(mask):
    mask[mask>0]=255
    return mask


def crop_save(frames, crop_datas, save_path):
    [
        *map(
            cv2.imwrite,
            [os.path.join(save_path, fn) for _,fn,_ in frames],
            map(crop_img, [frame for _,_,frame in frames], crop_datas),
        )
    ]
    pass


def parse_video(video, label, path, save_path, rela_path, faces_prefix, samples, face_scale, detector):
    raw_rela_path = os.path.join(rela_path,'raw','videos')
    c23_rela_path = os.path.join(rela_path,'c23','videos')
    c40_rela_path = os.path.join(rela_path,'c40','videos')
    gen_dirs(os.path.join(save_path, faces_prefix, raw_rela_path))
    gen_dirs(os.path.join(save_path, faces_prefix, c23_rela_path))
    gen_dirs(os.path.join(save_path, faces_prefix, c40_rela_path))
    raw_frames = list(video2frames(os.path.join(path, raw_rela_path, video), samples))  # orders, file_names, raw_frames
    crop_datas = [*map(partial(get_face_location, face_scale=face_scale, detector=detector), raw_frames[2])]
    # raw_frames = [(raw_frames[0][i],raw_frames[1][i],raw_frames[2][i]) for i in range(len(raw_frames[0])) if crop_datas[i] is not None]
    raw_frames[0] = [raw_frames[0][i] for i in range(len(raw_frames[0])) if crop_datas[i] is not None]
    raw_frames[1] = [raw_frames[1][i] for i in range(len(raw_frames[1])) if crop_datas[i] is not None]
    raw_frames[2] = [raw_frames[2][i] for i in range(len(raw_frames[2])) if crop_datas[i] is not None]
    crop_datas = [crop_data for crop_data in crop_datas if crop_data is not None]
    c23_frames = video2frames(os.path.join(path,c23_rela_path,video), samples)
    c40_frames = video2frames(os.path.join(path,c40_rela_path,video), samples)

    if label == '1':
        masks_rela_path = os.path.join(path,rela_path,'masks','videos',video)
        gen_dirs(os.path.join(save_path, masks_rela_path))
        masks_frames = list(video2frames(os.path.join(path,masks_rela_path,video), samples))
        final_orders = list(set(raw_frames[0]) & set(c23_frames[0]) & set(c40_frames[0]) & set(masks_frames[0]))
    else:
        final_orders = list(set(raw_frames[0]) & set(c23_frames[0]) & set(c40_frames[0]))
    crop_datas = [crop_data for i, crop_data in enumerate(crop_datas) if raw_frames[0][i] in final_orders]  # this err
    raw_frames = [(frame_ind, raw_frames[1][i], raw_frames[2][i]) for i, frame_ind in enumerate(raw_frames[0]) if frame_ind in final_orders]
    c23_frames = [(frame_ind, c23_frames[1][i], c23_frames[2][i]) for i, frame_ind in enumerate(c23_frames[0]) if frame_ind in final_orders]
    c40_frames = [(frame_ind, c40_frames[1][i], c40_frames[2][i]) for i, frame_ind in enumerate(c40_frames[0]) if frame_ind in final_orders]
    crop_save(raw_frames, crop_datas, os.path.join(save_path,faces_prefix,raw_rela_path))
    crop_save(c23_frames, crop_datas, os.path.join(save_path,faces_prefix,c23_rela_path))
    crop_save(c40_frames, crop_datas, os.path.join(save_path,faces_prefix,c40_rela_path))
    raw_infos = [os.path.join(raw_rela_path,fn)+f'\t{label}\n' for _,fn,_ in raw_frames]
    c23_infos = [os.path.join(c23_rela_path,fn)+f'\t{label}\n' for _,fn,_ in c23_frames]
    c40_infos = [os.path.join(c40_rela_path,fn)+f'\t{label}\n' for _,fn,_ in c40_frames]
    if label =='1':  # FaceShifter will ERR
        masks_frames[2] = [*map(bina_mask,masks_frames[2])]
        masks_frames = [(frame_ind, masks_frames[1][i], masks_frames[2][i]) for i, frame_ind in enumerate(masks_frames[0]) if frame_ind in final_orders]
        crop_save(masks_frames, crop_datas, os.path.join(save_path,faces_prefix,masks_rela_path))
        raw_infos = [info[:-1]+'\t'+os.path.join(masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(raw_infos)]
        c23_infos = [info[:-1]+'\t'+os.path.join(masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(c23_infos)]
        c40_infos = [info[:-1]+'\t'+os.path.join(masks_rela_path,masks_frames[i][1])+'\n' for i, info in enumerate(c40_infos)]
    return raw_infos, c23_infos, c40_infos

def write_infos(txt_path, infos):
    with open(txt_path, 'w') as f:
        f.writelines(infos)

def parse_split(mode, split, datasets, subset, faces_prefix, path, save_path, samples, face_scale, detector, num_workers):
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
            sub_split = [v for v in split if len(v)==7]
        elif 'DeepFakeDetection_original' == dataset:
            rela_path = os.path.join(
                'original_sequences', 'actors'
            )
            sub_split = [v for v in split if v[2:4]=='__']
        else:
            rela_path = os.path.join('manipulated_sequences', dataset)
            label = '1'
            sub_split = [v for v in split if not (len(v)==7 or v[2:4]=='__')]
        with mp.Pool(num_workers) as workers:
            with tqdm(total=len(sub_split)) as pbar:
                for raw_info, c23_info, c40_info in workers.imap_unordered(
                    partial(
                        parse_video,
                        label=label,
                        path=path,
                        save_path=save_path,
                        rela_path=rela_path,
                        faces_prefix=faces_prefix,
                        samples=samples,
                        face_scale=face_scale,
                        detector=detector,
                    ),
                    sub_split,
                ):
                    pbar.update()
                    raw_infos += raw_info
                    c23_infos += c23_info
                    c40_infos += c40_info
    assert len(raw_infos)==len(c23_infos)
    assert len(raw_infos)==len(c40_infos)
    print(mode, datasets, len(raw_infos))
    write_infos(os.path.join(save_path, faces_prefix, f'{subset}_{mode}_raw.txt'), raw_infos)
    write_infos(os.path.join(save_path, faces_prefix, f'{subset}_{mode}_c23.txt'), c23_infos)
    write_infos(os.path.join(save_path, faces_prefix, f'{subset}_{mode}_c40.txt'), c40_infos)
    return raw_infos, c23_infos, c40_infos


def main(subset, path, save_path, samples, face_scale, detector, num_workers):
    faces_prefix = f'faces{samples}{detector}'
    gen_dirs(os.path.join(save_path, faces_prefix))
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
    
    train_raw_infos, train_c23_infos, train_c40_infos = parse_split('train', train_split, datasets, subset, faces_prefix, path, save_path, samples, face_scale, detector, num_workers)
    val_raw_infos, val_c23_infos, val_c40_infos = parse_split('val', val_split, datasets, subset, faces_prefix, path, save_path, samples, face_scale, detector, num_workers)
    test_raw_infos, test_c23_infos, test_c40_infos = parse_split('test', test_split, datasets, subset, faces_prefix, path, save_path, samples, face_scale, detector, num_workers)

    write_infos(os.path.join(save_path,faces_prefix,f'{subset}_all_raw.txt'),train_raw_infos+val_raw_infos+test_raw_infos)
    write_infos(os.path.join(save_path,faces_prefix,f'{subset}_all_c23.txt'),train_c23_infos+val_c23_infos+test_c23_infos)
    write_infos(os.path.join(save_path,faces_prefix,f'{subset}_all_c40.txt'),train_c40_infos+val_c40_infos+test_c40_infos)


'''
change code 'return' to 'continue' to download the full datasets in download_ffdf.py
'''
# real is 0
# python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -save_path '/share/home/zhangchao/local_sets/ff++' -subset FF -samples 20 -scale 1.3 -detector dlib -workers 16
# python FaceForensics++.py -path '/share/home/zhangchao/datasets_io03_ssd/ff++' -save_path '/share/home/zhangchao/local_sets/ff++' -subset DFD -samples 20 -scale 1.3 -detector dlib -workers 16
if __name__ == '__main__':
    args = parse()
    main(args.subset, args.path, args.save_path, args.samples, args.scale, args.detector, args.workers)
