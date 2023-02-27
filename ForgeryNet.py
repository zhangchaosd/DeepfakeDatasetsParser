from functools import partial
import os
import multiprocessing as mp
from tqdm import tqdm
from utils import gen_dirs,read_txt,get_face_location,crop_img,parse
import cv2


def crop_save(img, crop_data, path, faces_prefix, rela_path, image_name):
    face = crop_img(img, crop_data)
    save_folder = os.path.join(path, faces_prefix, rela_path)
    gen_dirs(save_folder)
    face_name = image_name[:-3]+'png'
    cv2.imwrite(os.path.join(save_folder,face_name),face)
    return os.path.join(rela_path, face_name)

def parse_image(info, mode, image_folder, masks_folder, path, save_path, faces_prefix, face_scale, detector):
    image_name, bin_label, _, _ = info.split(' ')
    rela_path = os.path.join(mode,'image',os.path.dirname(image_name))
    image_name = os.path.basename(image_name)
    img = cv2.imread(os.path.join(path,rela_path, image_name))
    crop_data = get_face_location(img, face_scale, detector)
    if crop_data is None:
        print(f'Cannot found face {info}')
        return ''
    face_path = crop_save(img, crop_data, save_path, faces_prefix, rela_path, image_name)
    line = face_path+'\t'+bin_label + '\n'
    if bin_label == '1':
        mask_rela_path = rela_path.replace('image','spatial_localize').replace(image_folder,masks_folder)
        mask = cv2.imread(os.path.join(path, mask_rela_path, image_name),cv2.IMREAD_GRAYSCALE)
        mask_path = crop_save(mask, crop_data, save_path, faces_prefix, mask_rela_path, image_name)
        line = line[:-1]+'\t'+mask_path+'\n'
    return line

def parse_split(mode, image_folder, masks_folder, path, save_path, faces_prefix, face_scale, detector, num_workers):
    print(f'Parsing {mode}...')
    images = read_txt(os.path.join(path,mode,'image_list.txt'))
    infos = []
    with mp.Pool(num_workers) as workers:
        with tqdm(total=len(images)) as pbar:
            for info in workers.imap_unordered(
                partial(
                    parse_image,
                    mode=mode,
                    image_folder=image_folder,
                    masks_folder=masks_folder,
                    path=path,
                    save_path=save_path,
                    faces_prefix=faces_prefix,
                    face_scale=face_scale,
                    detector=detector,
                ),
                images,
            ):
                pbar.update()
                infos.append(info)
    with open(os.path.join(save_path,faces_prefix,f'{mode}.txt'),'w') as f:
        f.writelines(infos)
    print(mode, len(infos))
    return infos


def main(path, save_path, face_scale, detector, num_workers):
    faces_prefix = f'faces_{detector}'
    gen_dirs(os.path.join(save_path, faces_prefix))

    all_infos = []
    modes = [('Training', 'train_release', 'train_mask_release'), ('Validation', 'val_perturb_release', 'val_perturb_release')]
    for mode in modes:
        all_infos += parse_split(mode[0], mode[1], mode[2], path, save_path, faces_prefix, face_scale, detector, num_workers)
    with open(os.path.join(save_path,faces_prefix,'all.txt'),'w') as f:
        f.writelines(all_infos)
    print('All done', len(all_infos))


# python ForgeryNet.py -path '/share/home/zhangchao/datasets_io03_ssd/ForgeryNet' -scale 1.3 -detector dlib -workers 32
if __name__ == '__main__':
    args = parse()
    main(args.path, args.save_path, args.scale, args.detector, args.workers)