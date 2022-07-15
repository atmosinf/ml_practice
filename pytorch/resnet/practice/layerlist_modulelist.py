import torch
import torch.nn as nn

layers = []

layers.append(nn.Conv2d(3, 64, 3))
layers.append(nn.Conv2d(64, 64, 3))
layers.append(nn.Conv2d(64, 64, 3))

modlist = nn.ModuleList()
modlist.append(nn.Conv2d(3, 64, 3))
modlist.append(nn.Conv2d(64, 64, 3))
modlist.append(nn.Conv2d(64, 64, 3))

print('layerlist:\n', layers)
# print('list_to_sequential:\n', nn.Sequential(layers))
print('list_to_sequential_unpacked:\n', nn.Sequential(*layers))
print('modlist:\n', modlist)