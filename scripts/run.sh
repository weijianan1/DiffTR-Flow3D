CUDA_VISIBLE_DEVICES=0,1 python diffsf_main.py --train_dataset particle --val_dataset particle --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 \
        --result_dir results \

CUDA_VISIBLE_DEVICES=0,1 python diffsf_main.py --train_dataset particle --val_dataset particle --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 \
        --result_dir results \
        --resume model_best.pt \
        --eval \

