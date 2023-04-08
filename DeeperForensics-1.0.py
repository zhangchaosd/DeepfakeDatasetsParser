import os

from utils import gen_dirs, parse, parse_videos_mp, read_txt, video2face_pngs


def parse_one_video(video, path, save_path, label, faces_prefix, samples, face_scale, detector):
    rela_path = os.path.dirname(video)
    save_path = os.path.join(save_path, faces_prefix, rela_path)
    video_path = os.path.join(path, video)
    gen_dirs(os.path.join(path, faces_prefix, rela_path))
    infos = [
        os.path.join(rela_path, img_name) + f'\t{label}\n'
        for img_name in video2face_pngs(
            video_path, save_path, samples, face_scale, detector
        )
    ]
    return infos


def main(path, save_path, samples, face_scale, detector, num_workers):
    faces_prefix = f'faces{samples}{detector}'
    gen_dirs(os.path.join(save_path, faces_prefix))

    source_videos = read_txt(
        os.path.join(
            path, 'lists', 'source_videos_lists', 'source_videos_list.txt'
        )
    )
    manipulated_videos = read_txt(
        os.path.join(
            path, 'lists', 'manipulated_videos_lists', 'manipulated_videos_list.txt'
        )
    )
    modes = ['train', 'val', 'test']
    all_infos = []
    for mode in modes:
        print(f'Parsing {mode} split...')
        split_infos = read_txt(
           os.path.join(path, 'lists', 'splits', mode + '.txt')
        )
        split_infos = [info[4:8] for info in split_infos]
        s_videos = [video for video in source_videos if video[14:18] in split_infos]
        m_videos = [video for video in manipulated_videos if os.path.basename(video)[4:8] in split_infos]
        infos = parse_videos_mp(s_videos,'0',path,save_path,faces_prefix, samples, face_scale, detector, num_workers, f'Parsing {mode} source...', parse_one_video)
        infos += parse_videos_mp(m_videos,'1',path,save_path,faces_prefix, samples, face_scale, detector, num_workers, f'Parsing {mode} manipulated...', parse_one_video)
        with open(os.path.join(save_path, faces_prefix, f'{mode}.txt'), "w") as f:
            f.writelines(infos)
        all_infos += infos
    with open(os.path.join(save_path, faces_prefix, f'all.txt'), "w") as f:
        f.writelines(all_infos)
    print("All Done")


# 1 is fake
# python DeeperForensics-1.0.py -path '/share/home/zhangchao/datasets_io03_ssd/DeeperForensics-1.0' -save_path '/share/home/zhangchao/datasets_io03_ssd/DeeperForensics-1.02' -samples 20 -scale 1.3 -detector dlib -workers 32
if __name__ == '__main__':
    args = parse()
    main(args.path, args.save_path, args.samples, args.scale, args.detector, args.workers)

# 703,96,201   980_W010.mp4
# source 48475+189 source_videos/W136/light_up/surprise/camera_down/W136_light_up_surprise_camera_down.mp4
# fake 11000 manipulated_videos/end_to_end_mix_2_distortions/103_W018.mp4