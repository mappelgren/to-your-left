dataset=clevr-images-unambigous-dale-two

python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 10 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 10 --sender_embedding 10 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 10 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 50 --sender_embedding 50 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 30 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 10 --sender_embedding 10 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 13 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 50 --sender_embedding 50 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 20 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 50 --sender_embedding 50 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 100 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 10 --sender_embedding 10 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 100 --receiver_hidden 100 --sender_hidden 100 --receiver_embedding 100 --sender_embedding 100 --max_samples 10000 --model discriminator --n_epochs 10 --save
python source/mlt/language_games/play.py --scene_json_dir /scratch/guskunkdo/"$dataset"/scenes --image_dir /scratch/guskunkdo/"$dataset"/images --feature_file /scratch/guskunkdo/"$dataset"/features/bounding_box_resnet_4_avgpool_no_fc.h5 --vocab_size 100 --receiver_hidden 10 --sender_hidden 10 --receiver_embedding 50 --sender_embedding 50 --max_samples 10000 --model discriminator --n_epochs 10 --save
