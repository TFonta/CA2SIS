python ./train.py --name celeba_model --exclude_bg \
     --sample_dir ./samples/train_celeba/ --checkpoints_dir ./checkpoints/train_celeba/ \
    --batchSize 16 --crop_size 256 --style_dim 256 --load_size 256 --dataset_mode custom \
    --label_dir ./datasets/CelebA-HQ/train/labels \
    --image_dir ./datasets/CelebA-HQ/train/images \
    --label_dir_test ./datasets/CelebA-HQ/test/labels \
    --image_dir_test ./datasets/CelebA-HQ/test/images \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0
    