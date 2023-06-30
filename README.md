# DeepfakeDatasetsParser
Parse deepfake datasets to faces for traning and detection.
Can generate label text file and landmarks file.


## Results

The code will create a new folder starting with "faces_" in the root directory of the dataset: some folders, txt files and json files.


info txt format:
```
rela_path/real_face_0.png    0
rela_path/fake_face_0.png    1
"or"
rela_path/fake_face_0.png    1    rela_path/real_face_mask.png
```



## Installation


```
conda create --name dlib python=3.10
conda activate dlib
conda install -c https://conda.anaconda.org/conda-forge dlib
pip install opencv-python tqdm torch torchvision numpy imutils
```
<!-- For Retinaface:
```
pip install opencv-python retinaface-pytorch tqdm
``` -->

## Usage

Just run the corresponding py file for the dataset you want to process:

```
python xxxxx.py {your_dataset_path} {Number_of_frames_extracted_from_each_video}
```
The second parameter, if passed as 0, means that each frame is extracted.

For example:
```
python CelebDF.py 'xxx/Celeb-DF' 0
```
## Dataset Folder Structures

### UADFV

- /UADFV
  - /fake
    - 0000_fake.mp4
    - ...
  - /real
    - 0000.mp4
    - ...


### Celeb-DF v1 & v2

- /Celeb-DF
  - /Celeb-real
    - id0_0000.mp4
    - ...
  - /Celeb-synthesis
    - id0_id16_0000.mp4
    - ...
  - /YouTube-real
    - 00000.mp4
    - ...
  - List_of_testing_videos.txt

### FaceForensics++
- /FaceForensics++
  - /original_sequences
    - /youtube
      - /raw
        - /videos
          - 000.mp4
          - ...
      - /c23
        - ...
      - /c40
        - ...
  - /manipulated_sequences
    - /Deepfakes
      - /raw
        - /videos
          - 000_003.mp4
          - ...
      - /c23
        - ...
      - /c40
        - ...
      - /masks
        - ...
    - /Face2Face
      - ...
    - /FaceShifter    # no masks
      - ...
    - /FaceSwap
      - ...
    - /NeuralTextures
      - ...


### Deep Fake Detection Dataset

Joined FaceForensics++ and same as FaceForensics++.

### ForgeryNet
- /ForgeryNet
  - /Training
    - image_list.txt
    - /image
      - /train_release
        - /1
          - /7283059e2ee6ec302df2936c7d23313b
            - /fffdd2adada7489fa4628496f4b9b030
              - /frame00019.jpg
              - ...
            - ...
        - ...
    - /spatial_localize
      - /train_mask_release
        - /1
          - /7283059e2ee6ec302df2936c7d23313b
            - /fffdd2adada7489fa4628496f4b9b030
              - /frame00019.jpg
              - ...
            - ...
        - ...
  - /Validation
    - image_list.txt
    - /image
      - /val_perturb_release
        - ..
    - /spatial_localize
      - /val_perturb_release
        - ..
### DFDC

- /DFDC
  - /train
    - /dfdc_train_part_0
      - metadata.json
      - aaqaifqrwn.mp4
      - ...
    - ...
  - /validation
    - label.csv
    - 4000.mp4
    - ...
  - /test
    - metadata.json
    - label.csv
    - aalscayrfi.mp4
    - ...

### DeeperForensics-1.0
- /DeeperForensics-1.0
  - /splits
    - /splits
      - /train.txt
      - /val.txt
      - /test.txt
    - /source_videos_lists
      - ...
    - /manipulated_videos_lists
      - ...
    - /manipulated_videos_distortions_meta
      - ...
  - /source_videos
    - /M004
      - /BlendShape
        - /camera_down
          - M004_BlendShape_camera_down.mp4
        - ...
      - ...
    - ...
  - /manipulated_videos
    - /end_to_end
      - /000_M101.mp4
      - ...
    - ...
