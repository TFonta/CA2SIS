python ./test_style_mix.py --name celeba_model --exclude_bg \
    --sample_dir ./results/style_mix/ --checkpoints_dir ./checkpoints/train_celeba/ \
    --batchSize 4 --dataset_mode custom \
    --label_dir ./datasets/CelebA-HQ/train/labels \
    --image_dir ./datasets/CelebA-HQ/train/images \
    --label_dir_test ./datasets/CelebA-HQ/test/labels \
    --image_dir_test ./datasets/CelebA-HQ/test/images \
     --label_nc 19 --no_instance --nThreads 4 --gpu_ids 0 --part_to_swap makeup --swap_type style --att_loss
    
