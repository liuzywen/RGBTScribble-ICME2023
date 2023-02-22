import torch
from BBS_Model.pvtv2 import pvt_v2_b2

path = "D:\HXS\pretraining parameters\pvt_v2_b2.pth"
state = torch.load(path)
print(state.keys())
model = pvt_v2_b2()
print(model)
