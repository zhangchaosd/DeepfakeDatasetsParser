import sys
import os
from tqdm import tqdm
from utils import gen_dirs,read_txt,get_face_location,crop_img
import cv2

def bina_mask(mask):
    mask[mask>0]=255
    return mask

def parse(path, faces_path, rela_path, infos, face_scale):
#path /share/home/zhangchao/datasets_io03_ssd/ForgeryNet
#faces_path /share/home/zhangchao/datasets_io03_ssd/ForgeryNet/faces
#rela_path Training/image
#infos ['train_release/1/7283059e2ee6ec302df2936c7d23313b/62cfb9afdee4bf34bbfd9287a28e049a/frame00052.jpg 1 2 8']
    txt = []
    for info in tqdm(infos):
        pa, bin_label, _, _ = info.split(' ')
        img = cv2.imread(os.path.join(path,rela_path,pa))
        crop_data = get_face_location(img, face_scale)
        if crop_data is None:
            print('Cannot found face')
            continue
        face = crop_img(img, crop_data)
        face_folder = os.path.join(faces_path,rela_path,os.path.dirname(pa))
        gen_dirs(face_folder)
        #face_folder  /share/home/zhangchao/datasets_io03_ssd/ForgeryNet/faces/Training/image/train_release/1/7283059e2ee6ec302df2936c7d23313b/62cfb9afdee4bf34bbfd9287a28e049a
        cv2.imwrite(os.path.join(face_folder,os.path.basename(pa)),face,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        line = os.path.join('faces',rela_path,pa)+'\t'+bin_label + '\n'
        if bin_label == '1':
            mask_path= os.path.join(path,rela_path.replace('image','spatial_localize'),pa.replace('train_release','train_mask_release'))
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            mask = mask[crop_data[0]:crop_data[1],crop_data[2]:crop_data[3]]
            mask_folder = os.path.join(faces_path,rela_path.replace('image','spatial_localize'),os.path.dirname(pa).replace('train_release','train_mask_release'))
            gen_dirs(mask_folder)
            line = line[:-1]+'\t'+os.path.join('face',rela_path.replace('image','spatial_localize'),pa.replace('train_release','train_mask_release'))+'\n'
            cv2.imwrite(os.path.join(mask_folder,os.path.basename(pa)),mask,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # txt.append(os.path.join(rela_path,pa)+'\t'+bin_label+'\n')
        txt.append(line)
    return txt

def getsubset(na,i):
    if na[i+1]=='/':
        return na[i]
    return na[i:i+2]

def main(path, face_scale):
    subset = sys.argv[1]
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    txt = []

    train_info = os.path.join(path,'Training','image_list.txt')
    infos = read_txt(train_info)
    infos = [info for info in infos if getsubset(info,14)==subset]
    print('Parsing train data', subset)
    txt = parse(path,faces_path,os.path.join('Training','image'),infos,face_scale)
    with open(os.path.join(faces_path,str(subset)+'train.txt'),'w') as f:
        f.writelines(txt)
    print('train done')


    train_info = os.path.join(path,'Validation','image_list.txt')
    infos = read_txt(train_info)
    infos = [info for info in infos if getsubset(info,20)==subset]
    print('Parsing val data ', subset)
    txt = parse(path,faces_path,os.path.join('Validation','image'),infos,face_scale)
    with open(os.path.join(faces_path,str(subset)+'val.txt'),'w') as f:
        f.writelines(txt)
    print('val donne')



def read_txt(path):
    assert path.endswith('.txt')
    lines = []
    with open(path) as f:
        for line in f.readlines():
            lines.append(line.strip())
    return lines

# Call this func to merge txts in the end
def merge_txt(path):
    faces_path = os.path.join(path,'faces')
    train_info = []
    val_info = []
    for i in range(1,20):
        with open(os.path.join(faces_path,str(i)+'train.txt'),'r') as f:
            train_info = train_info + f.readlines()
        with open(os.path.join(faces_path,str(i)+'val.txt'),'r') as f:
            val_info = val_info + f.readlines()
    print('train: ', len(train_info))
    print('val: ', len(val_info))
    with open(os.path.join(faces_path,'train.txt'),'w') as f:
        f.writelines(train_info)
    with open(os.path.join(faces_path,'val.txt'),'w') as f:
        f.writelines(val_info)
    all_info = train_info + val_info
    print('all: ', len(all_info))
    with open(os.path.join(faces_path,'all.txt'),'w') as f:
        f.writelines(all_info)


if __name__ == '__main__':
    # path = '/share/home/zhangchao/datasets_io03_ssd/ForgeryNet'
    # merge_txt(path)
    # main(path, 1.3)
    exit()
    args = parse()
    main(args.path, args.samples, args.scale)