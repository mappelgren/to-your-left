python play.py --dataset_base_dir=/home/xappma/ --out_dir=out/ \
--validation_batch_size=32 --validation_batches_per_epoch=2 --n_epochs=51 --batch_size=32 \
--batches_per_epoch=100 --lr=0.0002 --validation_freq 3 --image_feature_file=resnet_3_noavgpool_no-fc.h5 \
--max_samples=100 --model=masked_attention_predictor --sender_cell=lstm --receiver_cell=lstm \
--save --sender_image_embedding=500 --sender_projection=100 --dataset dale-2 \
--sender_hidden 100 --receiver_hidden 10 --receiver_embeddin 10 --receiver_projection=100 \
--sender_embedding=15 --temperature 1 --vocab_size 16 --max_len 2

python play.py --dataset_base_dir=/home/xappma/ --out_dir=out/ \
--validation_batch_size=50 --validation_batches_per_epoch=20 --n_epochs=51 --batch_size=32 \
--batches_per_epoch=200 --lr=0.0002 --validation_freq 3 --image_feature_file=resnet_3_noavgpool_no-fc.h5 \
--max_samples=10000 --model=masked_attention_predictor --sender_cell=lstm --receiver_cell=lstm \
--save --sender_image_embedding=500 --sender_projection=100 --dataset dale-2 \
--sender_hidden 500 --receiver_hidden 500 --receiver_embeddin 10 --receiver_projection=100 \
--sender_embedding=10 --temperature 1 --vocab_size 16 --max_len 4



python play.py --dataset_base_dir=/home/xappma/ --out_dir=out/ \
--validation_batch_size=50 --validation_batches_per_epoch=20 --n_epochs=51 --batch_size=32 \
--batches_per_epoch=10 --lr=0.0002 --validation_freq 3 --image_feature_file=resnet_3_noavgpool_no-fc2.h5 \
--max_samples=1000 --model=masked_attention_predictor --sender_cell=lstm --receiver_cell=lstm \
--save --sender_image_embedding=500 --sender_projection=100 --dataset middle \
--sender_hidden 100 --receiver_hidden 100 --receiver_embeddin 100 --receiver_projection=100 \
--sender_embedding=15 --temperature 1 --vocab_size 16 --max_len 2

python play.py --dataset_base_dir=/home/xappma/ --out_dir=out/ \
--validation_batch_size=50 --validation_batches_per_epoch=20 --n_epochs=51 --batch_size=32 \
--batches_per_epoch=200 --lr=0.0002 --validation_freq 3 --image_feature_file=resnet_3_noavgpool_no-fc2.h5 --bounding_box_feature_file=resnet_3_noavgpool_no-fc2bb.h5 \
--max_samples=10000 --model=bounding_box_attention_predictor --sender_cell=lstm --receiver_cell=lstm \
--save --sender_image_embedding=500 --sender_projection=100 --dataset dale-2 \
--sender_hidden 500 --receiver_hidden 500 --receiver_embeddin 10 --receiver_projection=100 \
--sender_embedding=10 --temperature 1 --vocab_size 16 --max_len 4

