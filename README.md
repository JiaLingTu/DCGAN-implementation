# DCGAN-implementation



## Dataset
You can download dataset img align celeba.zip which was collected from the origin website [CelebFaces](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Requirements
The root folder should be structured as follows:
```
  root/
  ├─ img_align_celeba/  # you should download the dataset on the website and set the same name here.
  |  ├─ 000001.jpg
  |  ├─ 000002.jpg
  |  ├─ ...
  |  └─
  
  ├─ result/     # create after running main.py
  |
  ├─ main.py
  └─ utils.py
```

### Dependencies
```
matplotlib==3.5.1
numpy==1.22.1
Pillow==9.0.1
torch==1.10.1
torchvision==0.11.3
```

### Preprocess
|   |   |
|:--:|:--:|
|![GAN_Preprocessing.png](./image/GAN_Preprocessing.png)|![Dummy](./image/Dummy.png)|

Data先經過Resize後，短的那一邊被Resize成指定大小(在本題指定為64pixels),再用CenterCrop切出64×64的影像，在將影像轉換成Tensor之前，將RGB三個通道各別做Normalize到[-1, 1]之間，平均值為0.5。



