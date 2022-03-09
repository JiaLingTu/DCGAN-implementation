# DCGAN-implementation



## Dataset
You can download dataset img align celeba.zip which was collected from the origin website [CelebFaces] (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Requirements
The root folder should be structured as follows:
```
📁 root/
  ├─ 📁 img_align_celeba/
  |  ├─ 000001.jpg
  |  ├─ 000002.jpg
  |  ├─ ...
  |  └─
  
  ├─ 📁 result/     # create after running main.py
  |
  ├─ 📄 main.py
  ├─ 📄 train_disp.py
  ├─ 📄 train_norm.py
  └─ 📄 utils.py
```

### Dependencies
```
matplotlib==3.5.1
numpy==1.22.1
Pillow==9.0.1
torch==1.10.1
torchvision==0.11.3
```

## Results
Attach to the [report] (report.ipynb)


#### Parameters
Global parameters can be tinkered in the script:  
```python
DIR_TRAIN         # str, training dataset folder
DIR_VALID         # str, validation folder for output
DIR_TEST          # str, testing folder with custom images
CHK_OUT           # str, model checkpoint output folder
TEST_CROP         # int [px], center crop of custom testing images

# Training parameters
PARAMS = {
    "Type":       # str, just a name

    "pretrain":   # str or None, .pth filename in CHK_OUT folder to continue training, change to None to train from scratch

    "train": {
        "epochs":       # int, training epochs
        "batch":        # int, batch size
        "lr":           # float, learning rate
        "split":        # float 0~1, split percentage between training and validation dataset
        "nWorkers":     # int, Dataloader worker numbers
        "log_interv":   # int, epoch interval for valid image output
        },

    "image": {
        "img_resize":   # int [px], image resize size
        "img_crop":     # int [px], image center crop size
        "rand_flip":    # bool, flip training images randomly to add variation
        "rand_crop":    # int [px] or None, random crop in training image to add variation
        },

    "writer": False,    # bool, Tensorboard on/off
}
```
### How to import and use in Blender
Refer to the tutorial from the ambientcg help page: [Link](https://help.ambientcg.com/02-Using%20the%20assets/Using_PBR_maps_in_Blender.html)
