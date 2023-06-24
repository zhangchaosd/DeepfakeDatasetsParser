"""
1. Copy ./WildDeepfake_unzip.sh to your own WildDeepfake path.
2. run 'sh ./WildDeepfake_unzip.sh' to unzip all the .tar.gz files.
"""
import os
import sys


def parse_folder(dataset_path, folder):
    files = os.listdir(os.path.join(dataset_path, folder))
    files = [f.split(".")[0] for f in files if f != ".DS_Store"]
    files.sort(key=int)
    lines = []
    for file in files:
        lines.append(f"tar -xvf {folder}/{file}.tar.gz -C {folder}\n")
    with open(os.path.join(dataset_path, f"unzip_{folder}.sh"), "w") as f:
        f.writelines(lines)


def generate_unzip_sh(dataset_path):
    # This function is used for generating the unzip sh file
    parse_folder(dataset_path, "real_train")
    parse_folder(dataset_path, "real_test")


"""
Output should be:

train real videos: 371
squens: 3409
faces: 381876
train fake videos: 592
squens: 3099
faces: 632561
test real videos: 42
squens: 396
faces: 58659
test fake videos: 115
squens: 410
faces: 107003
"""


def parse_part(path, mode, label):
    bin_label = "0" if label == "real" else "1"
    videos = [
        os.path.join(f"{label}_{mode}", f, label)
        for f in os.listdir(os.path.join(path, f"{label}_{mode}"))
        if os.path.isdir(os.path.join(os.path.join(path, f"{label}_{mode}", f)))
    ]
    print(f"{mode} {label} videos: {len(videos)}")
    squens = [
        os.path.join(video, squen)
        for video in videos
        for squen in os.listdir(os.path.join(path, video))
    ]
    print(f"squens: {len(squens)}")
    infos = [
        os.path.join(squen, file) + "\t" + bin_label + "\n"
        for squen in squens
        for file in os.listdir(os.path.join(path, squen))
    ]
    print(f"faces: {len(infos)}")
    return infos


def parse_split(path, mode):
    infos = parse_part(path, mode, "real")
    infos += parse_part(path, mode, "fake")
    with open(os.path.join(path, f"{mode}.txt"), "w") as f:
        f.writelines(infos)
    return infos


def main(path):
    train_infos = parse_split(path, "train")
    assert len(train_infos) == 1014437
    test_infos = parse_split(path, "test")
    assert len(test_infos) == 165662
    with open(os.path.join(path, "all.txt"), "w") as f:
        f.writelines(train_infos + test_infos)


# 0 is real
# python WildDeepfake.py -path '/share/home/zhangchao/datasets_io03_ssd/WildDeepfake'
if __name__ == "__main__":
    # generate_unzip_sh(args.path)
    dataset_path = sys.argv[1]
    main(dataset_path)
