

from feature_extractors import extract_rotation_dataset
from train_rotation import run_game
import os
import configparser


def parse_feature_settings(feature_settings='DEFAULT'):
    config = configparser.ConfigParser()
    config.read('feature_setting.ini')
    config = config[feature_settings]

    c = {}

    for key in config.keys():

        if key in ['fc', 'bounding_boxes', 'avgpool']:
            c[key] = config.getboolean(key)
        elif key in ['batch_size', 'classifier_layers', 'num_blocks']:
            c[key] = config.getint(key)
        else:
            i = config.get(key)
            if i == 'None':
                i = None

            c[key] = i
    return c

def parse_game_settings(game_settings='DEFAULT'):
    config = configparser.ConfigParser()
    config.read('game_setting.ini')
    config = config[game_settings]

    c = {}

    for key in config.keys():

        if key in ['preemptable', 'no_cuda', 'tensorboard', 'fp16']:
            c[key] = config.getboolean(key)
        elif key in ['batch_size', 'batches_per_epoch', 'n_epochs',
                     'checkpoint_freq', 'validation_freq', 'update_freq',
                     'vocab_size', 'max_len', 'distributed_port', 'tau_s', 'game_size',
                     'hidden_size', 'embedding_size', 'featsize']:
            c[key] = config.getint(key)
        elif key in ['lr']:
            c[key] = config.getfloat(key)
        else:
            i = config.get(key)
            if i == 'None':
                i = None

            c[key] = i

    return c


def play(dataset, validation, feature_settings='DEFAULT', game_settings='DEFAULT'):

    feat = parse_feature_settings(feature_settings)
    game = parse_game_settings(game_settings)

    return _play(dataset, validation, **feat, **game)




def _play(dataset, validation, feature_dir='/home/xappma/to-your-left/data', device = 'cuda', feature_extractor = 'VGG', avgpool = False, classifier_layers = 3, num_blocks = 4, fc = False,
    bounding_boxes = False, scene_dir = 'scenes', batch_size = 32, image_dir = 'images', out_file = 'out', dataset_base_dir='/home/xappma/spatial-dataset',
         batches_per_epoch=30, mode='rf', n_epochs=10, random_seed=None,
         checkpoint_dir=None, preemptable=False, checkpoint_freq=0, validation_freq=1, load_from_checkpoint=None, no_cuda=False,
         optimizer='adam', lr=1e-2, update_freq=1, vocab_size=10, max_len=1, tensorboard=False, tensorboard_dir="runs/",
         distributed_port=18363, fp16=False, tau_s=10, game_size=2, hidden_size=20, embedding_size=50, featsize=1000
         ):

    feature_extraction_args = {'device':device, 'feature_extractor':feature_extractor, 'avgpool':avgpool,
                               'classifier_layers':classifier_layers, 'num_blocks':num_blocks, 'fc':fc,
                                'bounding_boxes':bounding_boxes, 'scene_dir':scene_dir,
                               'batch_size':batch_size, 'image_dir':image_dir, 'out_file':out_file}

    game_args = {'batch_size':batch_size, 'batches_per_epoch':batches_per_epoch, 'mode':mode, 'n_epochs':n_epochs, 'random_seed':random_seed, 'checkpoint_dir':checkpoint_dir,
    'preemptable':preemptable, 'checkpoint_freq':checkpoint_freq, 'validation_freq':validation_freq, 'load_from_checkpoint':load_from_checkpoint, 'no_cuda':no_cuda,
    'optimizer':optimizer, 'lr':lr, 'update_freq':update_freq, 'vocab_size':vocab_size, 'max_len':max_len, 'tensorboard':tensorboard, 'tensorboard_dir':tensorboard_dir,
    'distributed_port':distributed_port, 'fp16':fp16, 'tau_s':tau_s, 'game_size':game_size, 'hidden_size':hidden_size, 'embedding_size':embedding_size, 'featsize':featsize}

    train_dir = os.path.join(dataset_base_dir, dataset)
    name = extract_rotation_dataset(train_dir, feature_dir, **feature_extraction_args)

    if validation is not None:
        valid_dir = os.path.join(dataset_base_dir, validation)
        extract_rotation_dataset(valid_dir)

    run_game(root=feature_dir, dataset=dataset, features_name=name, validation_dataset=validation, **game_args)





