# DeepfakeDatasetParser
Parse deepfake datasets to faces for detection.
Can generate label text file.


## Installation

```
pip install opencv-python
pip install retinaface-pytorch
```

## Usage

```
python CelebDF.py -path xxx/xxx/Celeb-DF-v2 -samples 8 -scale 1.3
...
```
### params:
samples: Crop faces from each video.

sacle: The scale factor of bbox of face