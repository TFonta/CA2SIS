# CA2SIS

This code heavily borrows from the SEAN codebase https://github.com/ZPdesu/SEAN.

To run the pre-trained models and generate reconstructed images or train a new model, the user needs to download the CelebAMask-HQ dataset and put it inside a 'datasets' folder in the root directory.

Link to the CelebAMask-HQ dataset: https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view

To run the pre-trained models, please download the weights from this anonymous google drive https://drive.google.com/drive/folders/1Hvcm8epslPpehliyyr0I3HQHaXhvg-GV?usp=share_link and put the files into a 'checkpoints' folder in the root directory. (IMPORTANT: please note that the file timestamp in drive shows how the model checkpoints remained unchanged after the submission)

To train/test our final configuration, please leave the other options unchanged.

To test the model, simply run the bash script test.sh.
To train a new model, simply run the bash script train.sh.

The GPU memory occupancy for training is non-negligible, so we recommend using a small batch size if training on GPUs with reduced memory. 

### Train
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

### Test

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