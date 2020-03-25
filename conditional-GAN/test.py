import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import os

if not os.path.exists('./test'):
    os.mkdir('./test')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
# def to_img(x):
#     return x.view(-1, 1, 28, 28)
 

img_num = 16
z_dimension = 100
epoch = 10


# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(101, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 784), 
            nn.Tanh())
 
    def forward(self, x):
        x = self.gen(x)
        return x


def show(img):
    """ 
    用来显示图片的
    """
    plt.figure(figsize=(12, 8))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    
model = generator()
model.load_state_dict(torch.load(r'D:\program\PYTHON\GAN\GAN\代码\conditional-GAN\generator.pth'))
model.eval()
for _ in range(epoch):
    z = torch.randn(img_num, z_dimension)
    l = torch.randint(0,9,(img_num,1)).float()
    inputt = torch.cat((z,l),dim=1)
    out = model(inputt)

    outt = to_img(out.cpu().data)

    title = ''
    i = 0
    while(i<img_num):
        title = title + str(int(l[i][0]))
        i += 1

    save_image(outt, './test/{}.png'.format(title))