CUDA_VISIBLE_DEVICES=0,1 python difftr_flow3d_main.py --train_dataset TR-Flow3D --val_dataset TR-Flow3D --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 \
        --result_dir results

CUDA_VISIBLE_DEVICES=3 python difftr_flow3d_main.py --train_dataset TR-Flow3D --val_dataset TR-Flow3D --lr 4e-5 --train_batch_size 8 --test_batch_size 4 --num_epochs 500 \
        --result_dir results \
        --resume model_best.pt \
        --eval

