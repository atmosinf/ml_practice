'''
just a practice file
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random

# 1
# weight = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth')
# print('weight', weight)
# weight = {k: v.squeeze(0) if v.size(0) == 1 else v for k, v in weight.items()}
# # self.model.load_state_dict(weight)
# print('weight dict', weight)

# 2
# def l2_norm(x):
#     input_size = x.size()
#     buffer = torch.pow(x, 2)
#     print('buffer', buffer)
#     normp = torch.sum(buffer, 1).add_(1e-12)
#     print('normp', normp)
#     norm = torch.sqrt(normp)
#     print('norm', norm)
#     _output = torch.div(x, norm.view(-1, 1).expand_as(x))
#     output = _output.view(input_size)

#     return output

# inp = torch.randn(2,4) * 100
# print('inp', inp)
# print(l2_norm(inp))

# 3 - upsampling
ups = torch.randn((1,3,4,4)) * 10
print('input:', ups)
print('nearest:', nn.Upsample(scale_factor=2, mode='nearest')(ups))
# print('linear:', nn.Upsample(scale_factor=2, mode='linear')(ups))