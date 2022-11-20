import sys
import os
from tqdm import tqdm
# from utils import img2face,gen_dirs,read_txt
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
            txt.append(os.path.join(rela_path,pa)+'\t'+bin_label+'\n')
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
    with open(os.path.join(faces_path,str(subset)+'train.txt'),'w') as f:
        train_info = os.path.join(path,'Training','image_list.txt')
        infos = read_txt(train_info)
        infos = [info for info in infos if getsubset(info,14)==subset]
        print('Parsing train data', subset)
        txt = parse(path,faces_path,os.path.join('Training','image'),infos,face_scale)
        f.writelines(txt)
    print('train done')

    with open(os.path.join(faces_path,str(subset)+'val.txt'),'w') as f:
        train_info = os.path.join(path,'Validation','image_list.txt')
        infos = read_txt(train_info)
        infos = [info for info in infos if getsubset(info,20)==subset]
        print('Parsing val data ', subset)
        txt = parse(path,faces_path,os.path.join('Validation','image'),infos,face_scale)
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
    files = os.listdir(faces_path)
    train_txts = [f for f in files if f.endswith('train.txt')]
    val_txts = [f for f in files if f.endswith('val.txt')]
    assert len(train_txts) == 19 and len(val_txts)==19
    train_info = []
    for txt in train_txts:
        train_info = train_info + read_txt(os.path.join(faces_path, txt))
    val_info = []
    for txt in val_txts:
        val_info = val_info + read_txt(os.path.join(faces_path, txt))
    train_info = [l+'\n' for l in train_info]
    val_info = [l+'\n' for l in val_info]
    with open(os.path.join(faces_path,'train.txt'),'w') as f:
        f.writelines(train_info)
    with open(os.path.join(faces_path,'val.txt'),'w') as f:
        f.writelines(val_info)



if __name__ == '__main__':
    path = '/share/home/zhangchao/datasets_io03_ssd/ForgeryNet'
    merge_txt(path)
    # main('/share/home/zhangchao/datasets_io03_ssd/ForgeryNet', 1.3)
    exit()
    args = parse()
    main(args.path, args.samples, args.scale)