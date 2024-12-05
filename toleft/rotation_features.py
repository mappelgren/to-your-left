import os
from features import ImageNetFeat

import os
import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
import h5py
import argparse


class RotFeat(data.Dataset):
    def __init__(self, root='/home/xappma/to-your-left/data', dataset='one_colour_square',
                 name='VGG-pool_no-layers_3-fc_no-bb_no.h5py',  train=True, norm=None):
        import h5py

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        # FC features
        flat_features = []
        features = {}
        for target in ['target', 'distractor']:
            for rot in ['rot0', 'rot90', 'rot180', 'rot270']:
                fc_file = os.path.join(root, dataset, target, rot, name)
                fc = h5py.File(fc_file, 'r')
                key = list(fc.keys())[0]
                data = torch.FloatTensor(list(fc[key]))
                # img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
                # normed_data = data / img_norm
                # features[(target, rot)] = normed_data
                features[(target, rot)] = data

        # normalise data

        self.features = features
        self.create_obj2id(features)
        data = self.flat_features
        if norm is None:
            img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
        else:
            img_norm = norm
        normed_data = data / img_norm
        self.norm = img_norm
        self.flat_features = normed_data

        target_to_idx = {'target':1, 'distractor':0}
        rotation_to_idx = {'rot0':0, 'rot90':1, 'rot180':2, 'rot270':3}
        
        for t, r in features.keys():

            self.features[(t,r)] = self.flat_features[self.obj2id[target_to_idx[t]][rotation_to_idx[r]]['ims']]

    def __getitem__(self, index):
        return self.flat_features[index], index

    def __len__(self):
        return self.flat_features.size(0)

    def create_obj2id(self, feature_dict):

        self.obj2id = {}
        keys = {}
        idx_label = 0
        flat_features = None
        for i, (target, rot) in enumerate(feature_dict.keys()):
            if rot == 'rot0':
                r = 0
            elif rot == 'rot90':
                r = 1
            elif rot == 'rot180':
                r = 2
            else:
                r = 3
            
            if target not in keys:
                if target == 'target':
                    key = 1
                else:
                    key = 0
                keys[target] = key
                self.obj2id[key] = [{'labels':(target, 'rot0'), 'ims':[]},
                                    {'labels':(target, 'rot90'), 'ims':[]}, 
                                    {'labels':(target, 'rot180'), 'ims':[]}, 
                                    {'labels':(target, 'rot270'), 'ims':[]}]

            end = idx_label + len(feature_dict[(target, rot)])
            self.obj2id[keys[target]][r]['ims'] = np.array(list(range(idx_label, end)))
            idx_label = end
            if flat_features is None:
                flat_features = feature_dict[(target, rot)]
            else:
                flat_features = torch.cat((flat_features, feature_dict[(target, rot)]))
        self.flat_features = flat_features



class _BatchIterator:
    def __init__(self, loader, n_batches, seed=None):
        self.loader = loader
        self.n_batches = n_batches
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated > self.n_batches:
            raise StopIteration()

        batch_data = self.get_batch()
        self.batches_generated += 1
        return batch_data

    def get_batch(self):
        loader = self.loader
        opt = loader.opt

        # C = len(self.loader.dataset.obj2id.keys())  # number of concepts
        images_indexes_sender = np.zeros((opt.batch_size, opt.game_size))
        
        target_ims = loader.dataset.obj2id[1][0]['ims'] # get the target image with rotation 0
        distractor_ims = loader.dataset.obj2id[0][0]['ims']

        assert(target_ims[0] != distractor_ims[0])
                
        idxs = self.random_state.choice(list(range(len(target_ims))), opt.batch_size).astype(int)
        
        target = target_ims[idxs]
        distractor = distractor_ims[idxs]


        assert(target[0] != distractor[0])
        
        images_indexes_sender[:, 1] = target
        images_indexes_sender[:, 0] = distractor

        
        images_vectors_sender = []

        for i in range(opt.game_size):
            x = loader.dataset.flat_features[images_indexes_sender[:, i]]
            images_vectors_sender.append(x)

        images_vectors_sender = torch.stack(images_vectors_sender).contiguous()

        
        y = torch.zeros(opt.batch_size).long()

        
        images_vectors_receiver = torch.zeros_like(images_vectors_sender)
        for i in range(opt.batch_size):
            permutation = torch.randperm(opt.game_size)

            
            images_vectors_receiver[:, i, :] = images_vectors_sender[permutation, i, :]
            y[i] = permutation.argmin()

        return images_vectors_sender, y, images_vectors_receiver, {'image_ids':torch.tensor(idxs)}



class RotationLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.opt = kwargs.pop("opt")
        self.seed = kwargs.pop("seed")
        self.batches_per_epoch = kwargs.pop("batches_per_epoch")

        super(RotationLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _BatchIterator(self, n_batches=self.batches_per_epoch, seed=seed)
