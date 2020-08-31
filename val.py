import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import ASPDNet
import torch


from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root = '..'
#now generate the building's ground truth
building_train = os.path.join(root,'../train_data','images')
building_test = os.path.join(root,'../test_data','images')


path_sets = [building_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


model = ASPDNet()

model = model.cuda()

checkpoint = torch.load('model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
mae = 0
mse = 0
for i in range(len(img_paths)):
    file_path, filename = os.path.split(img_paths[i])
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    gt_count = np.sum(groundtruth)
    print(gt_count)
    with torch.no_grad():
        output = model(img.unsqueeze(0))
    pre_count = output.detach().cpu().sum().numpy()
    mae += abs(pre_count-gt_count)
    mse += (pre_count - gt_count) * (pre_count - gt_count)

mae = mae/len(img_paths)
mse = np.sqrt(mse/len(img_paths))

print(mae)
print(mse)



