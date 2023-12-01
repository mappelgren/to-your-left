import torch


def pixel_loss(model_output, ground_truth):
    loss = torch.diagonal(torch.cdist(model_output, ground_truth.float()))

    return torch.mean(loss)


def attention_loss(model_output, ground_truth):
    number_regions = 7
    print(model_output)
    print(model_output.max())
    model_x, model_y = to_coord(torch.max(model_output).indices, number_regions)
    print(model_output, model_x, model_y)
    truth_x, truth_y = to_coord(ground_truth, number_regions)
    loss = 0


def to_coord(region, number_regions):
    print(region)
    x = region % number_regions
    y = (region - x) / number_regions

    return x, y
