import os

from tqdm import tqdm

from utils import *

### TODO del
def dfs(pa = '/Users/zhangchao/datasets/ff'):
    if os.path.isdir(pa):
        files = os.listdir(pa)
        print(pa, len(files))
        files = [os.path.join(pa, f) for f in files]
        for f in files:
            dfs(f)
    else:
        if pa.endswith('mp4'):
            pass
        else:
            print('To remove', pa)
            # os.remove(pa)

pa = '/Users/zhangchao/datasets/ff'
# dfs(pa)

### TODO del

def get_source_videos(videos):
    res = []
    for video in videos:
        res.append(video[:3]+'.mp4')
        res.append(video[4:])
    return res

def double_videos(videos):
    res = []
    for video in videos:
        res.append(video[4:7]+'_'+video[:3]+'.mp4')
    return res + videos

def get_ff_splits(path):
    videos = [video for video in get_files_from_path(path) if int(video[:3])<int(video[4:7])]
    assert len(videos)==500, f'Missing some videos(expect 1000): {path}'
    static_shuffle(videos)
    train_split_m = videos[:360]
    val_split_m = videos[360:430]
    test_split_m = videos[430:]

    train_split_o = get_source_videos(train_split_m)
    val_split_o = get_source_videos(val_split_m)
    test_split_o = get_source_videos(test_split_m)

    train_split_m = double_videos(train_split_m)
    val_split_m = double_videos(val_split_m)
    test_split_m = double_videos(test_split_m)

    return train_split_m + train_split_o, val_split_m + val_split_o, test_split_m + test_split_o


def solve(dataset_path, faces_path, video_path, video_name, f, label, samples, face_scale, file_names = None, crop_datas = None):
    frames_with_file_names = [*video2frames(os.path.join(dataset_path,video_path,video_name),samples)]
    frames = [frame for _, frame in frames_with_file_names]
    if file_names is None:
        file_names = [file_name for file_name, _ in frames_with_file_names]
    file_names = [os.path.join(faces_path, video_path, file_name)for file_name, _ in frames_with_file_names]
    if crop_datas is None:
        crop_datas = [*map(get_face_location, frames, [face_scale]*len(frames))]
    # print(len(frames), len(file_names), len(crop_datas))
    assert len(frames) == len(file_names), f'{len(frames) != {len(file_names)}}'
    assert len(crop_datas) == len(file_names), f'{len(crop_datas) != {len(file_names)}}'
    faces = [*map(crop_img, frames, crop_datas)]
    [*map(cv2.imwrite, file_names, faces, [[int(cv2.IMWRITE_JPEG_QUALITY),100]]*len(faces))]
    if f is not None:
        f.writelines([os.path.join('faces', video_path, os.path.basename(img_name))+' '+label+'\n' for img_name in file_names])
    return file_names, crop_datas


def main(path, samples, face_scale, subset):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)

    f_train_raw = open(os.path.join(faces_path, 'train_raw.txt'), 'w')
    f_val_raw = open(os.path.join(faces_path, 'val_raw.txt'), 'w')
    f_test_raw = open(os.path.join(faces_path, 'test_raw.txt'), 'w')
    f_train_c23 = open(os.path.join(faces_path, 'train_c23.txt'), 'w')
    f_val_c23 = open(os.path.join(faces_path, 'val_c23.txt'), 'w')
    f_test_c23 = open(os.path.join(faces_path, 'test_c23.txt'), 'w')
    f_train_c40 = open(os.path.join(faces_path, 'train_c40.txt'), 'w')
    f_val_c40 = open(os.path.join(faces_path, 'val_c40.txt'), 'w')
    f_test_c40 = open(os.path.join(faces_path, 'test_c40.txt'), 'w')

    ### Attention
    train_split, val_split, test_split = get_ff_splits(os.path.join(path,'manipulated_sequences','Deepfakes','raw','videos'))

    # FaceShifter don't have masks
    if subset == 'FF':
        datasets = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    else:
        datasets = ['DeepFakeDetection_original', 'DeepFakeDetection']
    for i, dataset in enumerate(datasets):
        print(f'Now parsing {dataset}...')
        label = '0 '
        if 'original' == dataset:
            raw_path = os.path.join('original_sequences', 'youtube', 'raw', 'videos')
        elif 'DeepFakeDetection_original' == dataset:
            raw_path = os.path.join('original_sequences', 'actors', 'raw', 'videos')
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
        for video in tqdm(raw_videos):
            if video in train_split:
                f_raw = f_train_raw
                f_c23 = f_train_c23
                f_c40 = f_train_c40
            elif video in val_split:
                f_raw = f_val_raw
                f_c23 = f_val_c23
                f_c40 = f_val_c40
            else:
                f_raw = f_test_raw
                f_c23 = f_test_c23
                f_c40 = f_test_c40

            file_names, crop_datas = solve(path, faces_path,raw_path,video,f_raw,label,samples,face_scale)
            solve(path, faces_path,c23_path,video,f_c23,label,samples,face_scale,file_names,crop_datas)
            solve(path, faces_path,c40_path,video,f_c40,label,samples,face_scale,file_names,crop_datas)
            if 'original' not in dataset:
                solve(path, faces_path,masks_path,video,None,None,samples,face_scale,file_names,crop_datas)

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
change code 'return' to 'continue' download the full datasets in download_ffdf.py
'''

if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale, args.subset)
