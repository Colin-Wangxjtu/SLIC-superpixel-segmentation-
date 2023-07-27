from math import inf
import numpy as np
from numpy import float32, sqrt
import torch
import cv2



def get_RGB(path):
    img = cv2.imread(path)
    return torch.transpose(torch.tensor(img), 0, 1)

class SLIC():
    def __init__(self, h, w, k, m, feature, device): # 这里的k不一定为最后的k的数值，因为要开方、除等操作
        self.device = device
        self.h = h # 1024
        self.w = w # 750
        self.m = m
        local = torch.cat([torch.arange(self.h).view(-1, 1).unsqueeze(1).expand(h, w, 1), torch.arange(self.w).view(-1, 1).unsqueeze(0).expand(h, w, 1)], dim=-1)
        self.feature = torch.cat((local, torch.tensor(feature)), dim=2).to(self.device)

        self.label = torch.full((h, w), -1).to(self.device)
        self.distance = torch.full((h, w), float(inf)).to(self.device)
        center_h = h // k**0.5
        center_w = w // k**0.5
        center_h_num = int(h / center_h)
        center_w_num = int(w / center_w)
        self.k = center_h_num * center_w_num # k可能与输入不同
        self.center = torch.zeros((center_h_num*center_w_num, 5)).to(self.device) # 每一个聚类中心有5个属性，分别为x、y与5个属性
        self.s = int((h * w // self.k)**0.5)
        for i in range(center_h_num): # 初始化生成聚类中心，此时先不考虑移动到梯度最小地方的问题
            for j in range(center_w_num):
                self.center[i*center_w_num + j][0:2] = torch.tensor((int((i+0.5)*center_h), int((j+0.5)*center_w)))
        
    def kmeans(self, times):
        for time in range(times):
            for n, center in enumerate(self.center):
                x_min = int(max(0, center[0] - self.s))
                x_max = int(min(self.h, center[0] + self.s))
                y_min = int(max(0, center[1] - self.s))
                y_max = int(min(self.w, center[1] + self.s))
                search_space = self.feature[x_min: x_max, y_min: y_max]
                center_distance = distance(search_space, center, self.s, self.m)
                search_space_distance = self.distance[x_min: x_max, y_min: y_max]
                nearer = center_distance < search_space_distance # 生成一个bool的tensor
                search_space_distance[nearer] = center_distance[nearer]
                self.distance[x_min: x_max, y_min: y_max] = search_space_distance
                search_space_label = self.label[x_min: x_max, y_min: y_max]
                search_space_label[nearer] = n
                self.label[x_min: x_max, y_min: y_max] = search_space_label
            for n in range(self.k):
                label_n = self.label == n # 可生成bool张量
                super_pixel = self.feature[label_n]
                self.center[n] = torch.mean(super_pixel.float(), dim=0)
            print('epoch{} finish'.format(time))
        return self.center, self.label

def distance(search_space, center, s, m):
    space_d = ((search_space[:, :, 0] - center[0])**2 + (search_space[:, :, 1] - center[1])**2)**0.5
    color_d = ((search_space[:, :, 2] - center[2])**2 + (search_space[:, :, 3] - center[3])**2 + (search_space[:, :, 4] - center[4])**2)**0.5
    return ((color_d / m)**2 + (space_d / s)**2)**0.5
        
# hyper-parameters
m = 2
epochs = 10
k = 100

filename = 'photo6'
RGB_path = r'data/'+filename+'.jpg'
data_RGB = get_RGB(RGB_path)
h = data_RGB.shape[0]
w = data_RGB.shape[1]
SLIC_net = SLIC(h, w, k, m, data_RGB, 'cuda')
center, label = SLIC_net.kmeans(epochs)
new = torch.zeros((h, w, 3))
for n,c in enumerate(center):
    one_label = label == n
    new[one_label] = c[2:].to('cpu')
new = torch.transpose(new, 0, 1)
out = new.int().numpy().astype(np.uint8)
cv2.imshow('123', out)
cv2.waitKey(0)
cv2.imwrite('output/'+filename+'.jpg', out)
