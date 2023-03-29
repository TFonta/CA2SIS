python ./test.py --name ade20k_model \
    --dataset_mode ade20k --sample_dir ./samples/train_ade20k/ \
    --checkpoints_dir ./checkpoints/train_celeba/ \
    --batchSize 8 \
    --dataroot ./datasets/ADE20K --no_instance --nThreads 4 --gpu_ids 0