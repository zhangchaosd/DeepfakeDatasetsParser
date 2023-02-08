# DeepfakeDatasetsParser
Parse deepfake datasets to faces for detection.
Can generate label text file.


## Installation

For dlib:

```
conda create --name dlib python=3.10
conda install -c https://conda.anaconda.org/conda-forge dlib
pip install opencv-python tqdm
```
For Retinaface:
```
pip install opencv-python retinaface-pytorch tqdm
```

## Usage

```
python CelebDF.py -path 'xxx/Celeb-DF' -save_path 'xxx/Celeb-DF' -samples 20 -scale 1.3 -detector dlib -workers 8
...
```
### params:

TODO

samples: Crop faces from each video.

sacle: The scale factor of bbox of face