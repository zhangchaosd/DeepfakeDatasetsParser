import os
from utils import img2face,gen_dirs,read_txt
import cv2

def parse(path, faces_path, rela_path, infos, face_scale):
    txt = []
    for info in infos:
        pa, bin_label, _, _ = info.split(' ')
        img = cv2.imread(os.path.join(path,rela_path,pa))
        face = img2face(img,face_scale)
        if face:
            cv2.imwrite(os.path.join(faces_path,rela_path,pa),face,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            txt.append(os.path.join(rela_path,pa)+' '+bin_label+'\n')
    return txt


def main(path, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)
    with open(os.path.join(faces_path,'train.txt'),'w') as f:
        train_info = os.path.join(path,'Training','image_list.txt')
        infos = read_txt(train_info)
        txt = parse(path,faces_path,os.path.join('Training','image'),infos,face_scale)
        f.writelines(txt)
    with open(os.path.join(faces_path,'val.txt'),'w') as f:
        train_info = os.path.join(path,'Validation','image_list.txt')
        infos = read_txt(train_info)
        txt = parse(path,faces_path,os.path.join('Validation','image'),infos,face_scale)
        f.writelines(txt)


if __name__ == '__main__':
    main('/share/home/zhangchao/datasets_io03_ssd/ForgeryNet', 1.3)
    exit()
    args = parse()
    main(args.path, args.samples, args.scale)