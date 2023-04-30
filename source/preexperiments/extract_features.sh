#!/bin/bash
image_dir=$1
out_dir=$2
batch_size=$3
device=$4

python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_1_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 1 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_2_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 2 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_3_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 3 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_no-avgpool_no-fc.h5 --feature_extractor ResNet --no-avgpool --no-fc --num_blocks 4 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_avgpool_no-fc.h5 --feature_extractor ResNet --avgpool --no-fc --num_blocks 4 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"resnet_4_avgpool_fc.h5 --feature_extractor ResNet --avgpool --fc --num_blocks 4 --device $device --batch_size $batch_size

python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_no-avgpool_0.h5 --feature_extractor VGG --no-avgpool --classifier_layers 0 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_0.h5 --feature_extractor VGG --avgpool --classifier_layers 0 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_1.h5 --feature_extractor VGG --avgpool --classifier_layers 1 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_2.h5 --feature_extractor VGG --avgpool --classifier_layers 2 --device $device --batch_size $batch_size
python source/preexperiments/feature_extractors.py --image_dir $image_dir --out_file "$out_dir"vgg_avgpool_3.h5 --feature_extractor VGG --avgpool --classifier_layers 3 --device $device --batch_size $batch_size