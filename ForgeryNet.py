import sys
import os
from tqdm import tqdm
from utils import img2face,gen_dirs,read_txt
import cv2

def parse(path, faces_path, rela_path, infos, face_scale):
    txt = []
    for info in tqdm(infos):
        pa, bin_label, _, _ = info.split(' ')
        img = cv2.imread(os.path.join(path,rela_path,pa))
        face = img2face(img,face_scale)
        if face is not None:
            folder = os.path.join(faces_path,rela_path,os.path.dirname(pa))
            gen_dirs(folder)
            # print(os.path.join(faces_path,rela_path,pa),face.shape)
            cv2.imwrite(os.path.join(faces_path,rela_path,pa),face,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            txt.append(os.path.join(rela_path,pa)+' '+bin_label+'\n')
            # break
    return txt

def getsubset(na,i):
    if na[i+1]=='/':
        return na[i]
    return na[i:i+2]

def main(path, face_scale):
    subset = sys.argv[1]
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    with open(os.path.join(faces_path,'train.txt'),'w') as f:
        train_info = os.path.join(path,'Training','image_list.txt')
        infos = read_txt(train_info)
        infos = [info for info in infos if getsubset(info,14)==subset]
        print('Parsing train data', subset)
        txt = parse(path,faces_path,os.path.join('Training','image'),infos,face_scale)
        f.writelines(txt)
    print('train done')

    with open(os.path.join(faces_path,'val.txt'),'w') as f:
        train_info = os.path.join(path,'Validation','image_list.txt')
        infos = read_txt(train_info)
        infos = [info for info in infos if getsubset(info,20)==subset]
        print('Parsing val data ', subset)
        txt = parse(path,faces_path,os.path.join('Validation','image'),infos,face_scale)
        f.writelines(txt)
    print('val donne')


if __name__ == '__main__':
    main('/share/home/zhangchao/datasets_io03_ssd/ForgeryNet', 1.3)
    exit()
    args = parse()
    main(args.path, args.samples, args.scale)