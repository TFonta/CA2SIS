python ./test.py --name cityscapes_model \
    --dataset_mode cityscapes --sample_dir ./samples/train_cityscapes/ \
    --checkpoints_dir ./checkpoints/train_cityscapes/ \
    --batchSize 4 --style_enc_feat_dim 8  \
    --dataroot ./datasets/cityscapes --no_instance --nThreads 4 --gpu_ids 0 