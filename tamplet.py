import os
from utils import *


def main(path, samples, face_scale):
    faces_path = os.path.join(path, 'faces')
    gen_dirs(faces_path)


    pass


if __name__ == '__main__':
    main('/Users/zhangchao/datasets/UADFV', 2, 1.3)
    exit()
    args = parse()
    main(args.path, args.samples, args.scale)