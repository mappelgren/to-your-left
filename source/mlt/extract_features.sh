#!/bin/bash
image_dir=$1
out_dir=$2
batch_size=$3
device=$4

#python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_1_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 1 --device $device --batch_size $batch_size
#python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_2_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 2 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_3_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 3 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 4 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_avgpool_no-fc.h5 --feature_extractor ResNet --avgpool --no-fc --num_blocks 4 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_avgpool_fc.h5 --feature_extractor ResNet --avgpool --fc --num_blocks 4 --device $device --batch_size $batch_size

python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_no-avgpool_0.h5 --feature_extractor VGG --no-avgpool --classifier_layers 0 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_0.h5 --feature_extractor VGG --avgpool --classifier_layers 0 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_1.h5 --feature_extractor VGG --avgpool --classifier_layers 1 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_2.h5 --feature_extractor VGG --avgpool --classifier_layers 2 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_3.h5 --feature_extractor VGG --avgpool --classifier_layers 3 --device $device --batch_size $batch_size

python source/preexperiments/feature_extractors.py --image_dir /scratch/guskunkdo/clevr-images-unambigous-colour/images/ --scene_dir /scratch/guskunkdo/clevr-images-unambigous-colour/scenes/ --out_file /scratch/guskunkdo/clevr-images-unambigous-colour/features/bounding_box_resnet_4_avgpool_no-fc.h5 --feature_extractor ResNet --avgpool --no-fc --num_blocks 4 --device cuda --batch_size 200 --bounding_boxes

python feature_extractors.py --image_dir ~/one_middle_dataset/images/ --scene_dir ~/one_middle_dataset/scenes/ --out_file ~/one_middle_dataset/features/resnet_3_noavgpool_no-fc2.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 3 --device cuda --batch_size 32

python feature_extractors.py --image_dir ~/clevr-images-unambigous-dale-two/images/ --scene_dir ~/clevr-images-unambigous-dale-two/scenes/ --out_file ~/clevr-images-unambigous-dale-two/features/resnet_3_noavgpool_no-fc2bb.h5 --feature_extractor ResNet ---bounding_boxes -no-avgpool --no-fc --num_blocks 3 --device cuda --batch_size 30



python feature_extractors.py --image_dir ~/one_middle_dataset/images/ --scene_dir ~/one_middle_dataset/scenes/ --out_file ~/one_middle_dataset/features/resnet_3_noavgpool_no-fc2-bb.h5 --feature_extractor ResNet --bounding_boxes --no-avgpool --no-fc --num_blocks 3 --device cuda --batch_size 32

python feature_extractors.py --image_dir ~/spatial-dataset/two_mirrored_square/images/rot0 --scene_dir ~/spatial-dataset/two_mirrored_square/scenes --out_file ~/spatial-dataset/two_mirrored_sqare/features/vgg-fc.h5 --feature_extractor VGG --no-avgpool --fc --num_blocks 4 --device cuda --batch_size 32



python language_games/play.py --dataset_base_dir=/home/xappma/ --out_dir=out/ --validation_batch_size=32 --validation_batches_per_epoch=2 --n_epochs=50 --batch_size=32 --batches_per_epoch=1 --lr=0.0002 --validation_freq 3 --image_feature_file=resnet_3_noavgpool_no-fc.h5 --max_samples=1000 --model=masked_attention_predictor --sender_cell=lstm --receiver_cell=lstm --save --sender_image_embedding=500 --sender_projection=100 --dataset dale-2 --sender_hidden 100 --receiver_hidden 10 --receiver_embeddin 10 --receiver_projection=100 --sender_embedding=15 --temperature 1 --vocab_size 16 --max_len 2


python feature_extractors.py --image_dir ~/spatial-dataset/two_mirrored_square/images/distractor/rot0 --scene_dir ~/spatial-dataset/two_mirrored_square/target/scenes --out_file ~/spatial-dataset/features/vgg-fc-distractor-rot0.h5 --feature_extractor VGG --no-avgpool --fc --num_blocks 4 --device cuda --batch_size 32
