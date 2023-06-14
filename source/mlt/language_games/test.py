import torch

l = [torch.tensor(e) for e in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]

t = torch.stack(l)
t = t.view(2, 5)

print(t)
print(t.shape)
