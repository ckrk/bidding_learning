import torch


dtype = torch.float
device = torch.device("cpu")
# Activate GPU
device = torch.device("cuda:0") 

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

print(torch.cuda.is_available)