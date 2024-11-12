
for t in target distractor
do
  for r in rot0 rot90 rot180 rot270
  do
    echo $t $r
    python feature_extractors.py --image_dir ~/spatial-dataset/two_mirrored_square/images/$t/$r \
--scene_dir ~/spatial-dataset/two_mirrored_square/$t/scenes --out_file \
~/spatial-dataset/features/new-vgg-layers-2-$t-$r.h5 --feature_extractor VGG --no-avgpool \
--device cuda --batch_size 32 --classifier_layers 2
  done
done


