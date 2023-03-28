import sys

import torch
from classification_models import ResnetFeatureClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import MulticlassAccuracy

from data_readers import ClassifierDataset

if __name__ == '__main__':
    scenes_json, image_dir, max_samples, dev = sys.argv[1:]

    if dev == 'cpu':
        device = torch.device('cpu')
    elif dev == 'cuda':
        device = torch.device('cuda:0')
    else:
        raise AttributeError('Device must be cpu or cuda')
    
    dataset = ClassifierDataset(scenes_json, image_dir, max_samples)

    train_dataset_length = int(0.8 * len(dataset))
    test_dataset_length = len(dataset) - train_dataset_length
    train_dataset, test_dataset = random_split(dataset, (train_dataset_length, test_dataset_length))
    train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=1
        )
    test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=True, num_workers=1
        )
    
    model = ResnetFeatureClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(4):
        total_loss = 0
        for i, (model_input, ground_truth) in enumerate(train_loader):
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)

            output = model(model_input)

            loss = loss_function(output, ground_truth.long())

            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()

    model.eval()
    predictions = []
    ground_truths = []
    metric = MulticlassAccuracy()
    for (model_input, ground_truth) in enumerate(test_loader):
        model_input = model_input.to(device)
        ground_truth = ground_truth.to(device)
        output = model(model_input).detach()
        max_indices = torch.max(output, dim=1)[1]

        metric.update(max_indices, ground_truth)

    metric.compute()
    