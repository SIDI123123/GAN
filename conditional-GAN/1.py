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
import os

if not os.path.exists('./img1'):
    os.mkdir('./img1')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
# def to_img(x):
#     return x.view(-1, 1, 28, 28)
 

batch_size = 128
num_epoch = 500
z_dimension = 100


# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# MNIST dataset
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True)
# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size*2, shuffle=True)


# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(785, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x

 

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

 

D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

# Binary cross entropy loss and optimizer

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
 

# Start training

for epoch in range(num_epoch):
    for i, (img, label) in enumerate(dataloader):
        num_img = img.size(0)//2
        # =================train discriminator
        label = label.view(num_img*2,1).float()
        label = label[0:num_img]
        img = img.view(num_img*2, -1)
        unmatched_img = img[num_img:]
        img = img[0:num_img]
        inputt = torch.cat((img,label),dim=1)
        
        label_cuda = Variable(label).cuda()
        real_img = Variable(img).cuda()
        real_input = Variable(inputt).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

 

        # compute loss of real_img and matched
        real_out = D(real_input)
        d_loss_real_match = criterion(real_out, real_label)
        real_matched_scores = real_out  # closer to 1 means better

 

        # compute loss of fake_img
        z = torch.randn(num_img, z_dimension)
        z = torch.cat((z,label),dim=1)
        z = Variable(z).cuda()
        fake_img = G(z)
        fake_img = torch.cat((fake_img,label_cuda),dim=1)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

 
        #compute loss of real_img and unmatched
        unmatched_input = torch.cat((unmatched_img,label),dim=1)
        unmatched_input = Variable(unmatched_input).cuda()
        real_unmatched_out = D(unmatched_input)
        d_loss_real_unmatch = criterion(real_unmatched_out, fake_label)
        real_unmatched_scores = real_unmatched_out


        # bp and optimize
        d_loss = d_loss_real_match + d_loss_fake + d_loss_real_unmatch
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        # ===============train generator
        # compute loss of fake_img
        z = torch.randn(num_img, z_dimension)
        z = torch.cat((z,label),dim=1)
        z = Variable(z).cuda()
        fake_img = G(z)
        fake_img_input = torch.cat((fake_img,label_cuda),dim=1)
        output = D(fake_img_input)
        g_loss = criterion(output, real_label)
 

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

 

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real matchde: {:.6f}, D fake: {:.6f},D real unmatched: {:.6f}'.format(
                      epoch, num_epoch, d_loss.item(), g_loss.item(),
                      real_matched_scores.data.mean(), fake_scores.data.mean(), real_unmatched_scores.mean()))

    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './img/real_images.png')

 
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

 

torch.save(G.state_dict(), './generator(1).pth')
torch.save(D.state_dict(), './discriminator(1).pth')
