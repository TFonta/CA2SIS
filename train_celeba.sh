python3 ./train.py --name celeba_model --exclude_bg \
     --sample_dir ./samples/train_celeba_diversity/ --checkpoints_dir ./checkpoints/train_celeba_diversity/ \
    --batchSize 8 --dataset_mode custom \
    --label_dir ./datasets/CelebA-HQ/train/labels \
    --image_dir ./datasets/CelebA-HQ/train/images \
    --label_dir_test ./datasets/CelebA-HQ/test/labels \
    --image_dir_test ./datasets/CelebA-HQ/test/images \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0 \
     --pretrained_path ./checkpoints/pretrained_gen/ --att_loss --use_noise --use_ema \
    