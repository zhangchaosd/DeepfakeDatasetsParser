import os

from utils import gen_dirs, parse, read_txt, parse_videos_mp, video2face_pngs

modes = ['train', 'val', 'test']


def parse_videos(videos, get_key):
    d = {}
    for video in videos:
        key = get_key(video)
        if key in d:
            d[key].append(video)
        else:
            d[key] = [video]
    return d


def parse_one_video(video, path, label, faces_prefix, samples, face_scale, detector):
    rela_path = os.path.dirname(video)
    save_path = os.path.join(path, faces_prefix, rela_path)
    video_path = os.path.join(path, video)
    gen_dirs(os.path.join(path, faces_prefix, rela_path))
    infos = [
        os.path.join(faces_prefix, rela_path, img_name) + f'\t{label}\n'
        for img_name in video2face_pngs(
            video_path, save_path, samples, face_scale, detector
        )
    ]
    return infos


def main(path, samples, face_scale, detector, num_workers):
    faces_prefix = f'faces{samples}{detector}'
    gen_dirs(os.path.join(path, faces_prefix))

    source_videos = read_txt(
        os.path.join(
            path, 'lists', 'source_videos_lists', 'source_videos_list.txt'
        )
    )
    source_videos += read_txt(
        os.path.join(
            path, 'lists', 'source_videos_lists', 'source_videos_extra_list.txt'
        )
    )
    source_videos = parse_videos(source_videos, lambda s:s[14:18])
    manipulated_videos = read_txt(
        os.path.join(
            path, 'lists', 'manipulated_videos_lists', 'manipulated_videos_list.txt'
        )
    )
    manipulated_videos = parse_videos(manipulated_videos, lambda s:os.path.basename(s)[4:8])
    all_infos = []
    for mode in modes:
        print(f'Parsing {mode} split...')
        split_infos = read_txt(
           os.path.join(path, 'lists', 'splits', mode + '.txt')
        )
        s_videos = []
        m_videos = []
        for info in split_infos:
            s_videos += source_videos[info[4:8]]
            m_videos += manipulated_videos[info[4:8]]
        infos = parse_videos_mp(s_videos,'0',path,faces_prefix, samples, face_scale, detector, num_workers, f'Parsing {mode} source...', parse_one_video)
        infos += parse_videos_mp(m_videos,'1',path,faces_prefix, samples, face_scale, detector, num_workers, f'Parsing {mode} manipulated...', parse_one_video)
        with open(os.path.join(path, faces_prefix, f'{mode}.txt')) as f:
            f.writelines(infos)
        all_infos += infos
    with open(os.path.join(path, faces_prefix, 'all.txt')) as f:
        f.writelines(infos)


# 1 is fake
# python DeeperForensics-1.0.py -path '/share/home/zhangchao/datasets_io03_ssd/DeeperForensics-1.0' -samples 120 -scale 1.3 -detector dlib -workers 8
if __name__ == '__main__':
    args = parse()
    main(args.path, args.samples, args.scale, args.detector, args.workers)

# 703,96,201   980_W010.mp4
# source 48475+189 source_videos/W136/light_up/surprise/camera_down/W136_light_up_surprise_camera_down.mp4
# fake 11000 manipulated_videos/end_to_end_mix_2_distortions/103_W018.mp4