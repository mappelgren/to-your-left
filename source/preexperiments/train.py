import argparse
import sys

import torch
from classification_models import (ResnetAttentionAttributeClassifier,
                                   ResnetAttentionClassifier)
from data_readers import AttentionAttributeDataset, AttentionDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_json_file", type=str, default=None, help="Path to the scene JSON file"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None, help="Path to the scene image dir"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="max samples to load"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="number of epochs"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cpu or cuda"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size"
    )
    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda:0')
    else:
        raise AttributeError('Device must be cpu or cuda')
    
    dataset = AttentionAttributeDataset(args.scene_json_file, args.image_dir, args.max_samples)

    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(dataset, (train_dataset_length, test_dataset_length))
    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
        )
    test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
        )
    
    model = ResnetAttentionAttributeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    classifier_loss = nn.CrossEntropyLoss()

    def pixel_loss(model_output, ground_truth):
        loss = torch.diagonal(torch.cdist(model_output, ground_truth.float()))

        return torch.mean(loss)
    
    mse_loss = nn.MSELoss()

    def bounding_box_accuracy(model):
        model.eval()
        metric = MulticlassAccuracy()
        for model_input, ground_truth in test_loader:
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)
            output = model(model_input).detach()
            max_indices = torch.max(output, dim=1)[1]

            metric.update(max_indices, ground_truth)
        return metric.compute()
    
    def pixel_accuracy(model):
        model.eval()
        metric = BinaryAccuracy(device=device)
        for model_input, ground_truth in test_loader:
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)
            output = model(model_input).detach()
            
            distances = torch.diagonal(torch.cdist(output, ground_truth.float()))
            positives = torch.where(distances < 20, distances, 0)
            metric.update(positives, torch.ones_like(positives))
        return metric.compute()

    print(f'Batches per epoch: {len(train_loader)}')
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (model_input, ground_truth) in enumerate(train_loader):
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)

            output = model(model_input)

            # print('Truth:', ground_truth)
            # print('Prediction:', output)
            loss = pixel_loss(output, ground_truth)
            # print(loss)
            # print()
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()
        print(f'Accuracy: {pixel_accuracy(model)}')
