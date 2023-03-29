# CA2-SIS: Semantic Image Synthesis with Class-adaptive Cross-attention

![image](./images/eyecatcher.png)
**Figure:** *Face images style and shape editing with CA2-SIS*

[[Paper](PUT-LINK)]

Semantic Image Synthesis with Class-adaptive Cross-attention (CA2-SIS) enables high reconstruction quality, style transfer from a reference image both at global (full) and class level, and automatic shape editing.

## Installation

## Datasets Preparation
To run the pre-trained models and generate reconstructed images or train a new model, the user needs to download the CelebAMask-HQ dataset and put it inside a 'datasets' folder in the root directory.

Link to the CelebAMask-HQ dataset: https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view

## Generating Images Using Pre-trained Models

To run the pre-trained models, please download the related weights nd put the files into a 'checkpoints' folder in the root directory:

[[CelebMask-HQ](https://drive.google.com/drive/folders/1Hvcm8epslPpehliyyr0I3HQHaXhvg-GV?usp=share_link)]
[[Ade20K (To be uploaded)](link)]
[[CityScapes (To be uploaded)](link)]

Then, run the following command:

```
python ./test.py --name celeba_model --exclude_bg \
    --sample_dir ./results/train_celeba/ --checkpoints_dir ./checkpoints/train_celeba/ \
    --batchSize 16 --crop_size 256 --style_dim 256 --load_size 256 --dataset_mode custom \
    --label_dir ./datasets/CelebA-HQ/train/labels \
    --image_dir ./datasets/CelebA-HQ/train/images \
    --label_dir_test ./datasets/CelebA-HQ/test/labels \
    --image_dir_test ./datasets/CelebA-HQ/test/images \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0
```  

To train/test our final configuration, please leave the other options unchanged.

To test the model, simply run the bash script test.sh.
To train a new model, simply run the bash script train.sh.

The GPU memory occupancy for training is non-negligible, so we recommend using a small batch size if training on GPUs with reduced memory. 

## Train New Models
```
python ./train.py --name celeba_model --exclude_bg \
     --sample_dir ./samples/train_celeba/ --checkpoints_dir ./checkpoints/train_celeba/ \
    --batchSize 16 --crop_size 256 --style_dim 256 --load_size 256 --dataset_mode custom \
    --label_dir ./datasets/CelebA-HQ/train/labels \
    --image_dir ./datasets/CelebA-HQ/train/images \
    --label_dir_test ./datasets/CelebA-HQ/test/labels \
    --image_dir_test ./datasets/CelebA-HQ/test/images \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0
```   

### License
All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**) The code is released for academic research use only.


### Citation

## Acknowledgments
This code heavily borrows from the SEAN codebase https://github.com/ZPdesu/SEAN. 


