import torch
import torch.nn as nn
import torchvision

import copy
import time
from os import mkdir
from os.path import exists

import matplotlib

import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from torch.nn import BatchNorm1d
from torchvision.utils import save_image
from d2l import torch as d2l   # 或 from d2l import mxnet as d2l
name = "v1"
image_size = [1,28,28]
#学习率
g_lr = 0.0003
d_lr = 0.0003
#epoch
num_epochs =100
#batch_size
batch_size = 64
#随机维度
latent_dim = 64
#正则项
weight_decay=0.0001
#优化器 betas
g_optimizer_betas = (0.5,0.999)
d_optimizer_betas = (0.5,0.999)
#数据路径
data_dir = 'D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_1\data\cylinder\cylinder'
#如果路径不存在则创建路径
out_put_dir = "./train_data/"
loss_dir = "./train_loss/"
model_dir = "./train_model/"
predict_dir = "./predict/"
use_gpu = torch.cuda.is_available()
if not exists(out_put_dir):
    mkdir(out_put_dir)
if not exists(f"{out_put_dir}/{loss_dir}"):
    mkdir(f"{out_put_dir}/{loss_dir}")
if not exists(f"{out_put_dir}/{model_dir}"):
    mkdir(f"{out_put_dir}/{model_dir}")
if not exists(f"{out_put_dir}/{predict_dir}"):
    mkdir(f"{out_put_dir}/{predict_dir}")
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


class Animator:
    """Docker/.py 可用：记录数据，最后导出 GIF"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale="linear", yscale="linear",
                 fmts=("-", "m--", "g-.", "r:"), figsize=(6, 4)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.fmts = fmts

        self.X = None
        self.Y = None
        self.frames = []  # 每一帧保存一次快照

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n

        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 记录当前帧数据（深拷贝）
        self.frames.append(([row[:] for row in self.X], [row[:] for row in self.Y]))

    def _config_axes(self, current_y=None):
        self.ax.set_xlabel(self.xlabel if self.xlabel else "")
        self.ax.set_ylabel(self.ylabel if self.ylabel else "")
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        if self.ylim is not None:
            ymin, ymax = self.ylim[0],self.ylim[1]
            if current_y:
                data_max = max(
                    max(series) for series in current_y if series
                )
                ymax = max(ymax, data_max)  # 数据更大时，自动扩展上限
            self.ax.set_ylim((ymin, ymax))

        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)
        if self.legend:
            self.ax.legend(self.legend, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)

    def _draw_frame(self,frame_idx):
        self.ax.cla()
        Xf, Yf = self.frames[frame_idx]
        for i, (xx, yy) in enumerate(zip(Xf, Yf)):
            fmt = self.fmts[i % len(self.fmts)]
            self.ax.plot(xx, yy, fmt)
        self._config_axes(current_y=Yf)
    def save(self, gif_path="train.gif", png_path="train.png", fps=20):
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")



        def update(frame_idx):
            self._draw_frame(frame_idx)
            return self.ax.lines

        # 1) 保存 GIF
        ani = FuncAnimation(
            self.fig, update, frames=len(self.frames), interval=1000 / fps, blit=False
        )
        ani.save(gif_path, writer=PillowWriter(fps=fps))



        plt.close(self.fig)
    def save_png(self, png_path="train.png"):
        # 2) 保存 PNG（最后一帧）
        self._draw_frame(len(self.frames) - 1)
        self.fig.savefig(png_path, dpi=200, bbox_inches="tight")
class Generator(nn.Module):
    """生成器"""
    def __init__(self,in_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.GELU(),
            nn.Linear(in_features=128,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.GELU(),
            nn.Linear(in_features=256,out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.GELU(),
            nn.Linear(in_features=512,out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024,out_features=torch.prod(torch.tensor(image_size),dtype=torch.int32)),
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        shape of z:(batch_size,z_dim) (batch_size,latent_dim)(batch_size,latent_dim)
        """
        output = self.model(z)
        image = output.reshape(z.shape[0],*image_size)
        return image
class Discriminator(nn.Module):
    """判别器"""
    def __init__(self,in_channel):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.GELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.GELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.GELU(),
            nn.Linear(in_features=16, out_features=1),
            nn.Sigmoid(),
        )
    def forward(self, image):
        """
        shape of image:(batch_size,z_dim) (batch_size,1*28*28)
        """
        prob = self.model(image.reshape(image.shape[0],-1))

        return prob
def train():
    #导入数据集
    dataset = torchvision.datasets.MNIST(root=data_dir,train=True,transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)), #x in [0,255]
        torchvision.transforms.ToTensor(), #x/255 [0,1]
        # #归一化 (x-0.5)/0.5  x->[-1,1]
        # torchvision.transforms.Normalize((0.5,), (0.5,))
        ]),
                                         download=True)
    #数据装载器
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
    #实例化generator
    generator = Generator(in_channel=latent_dim)
    #实例化Discriminator
    discriminator = Discriminator(in_channel=torch.prod(torch.tensor(image_size),dtype=torch.int32))
    #优化器
    g_optimizer = torch.optim.Adam(generator.parameters(),lr=g_lr,betas=g_optimizer_betas,weight_decay=weight_decay)
    d_optimizer = torch.optim.Adam(discriminator.parameters(),lr=d_lr,betas=d_optimizer_betas,weight_decay=weight_decay)
    #损失函数 二叉交叉熵loss
    loss_func = nn.BCELoss()
    labels_one = torch.ones(batch_size, 1)
    labels_zero = torch.zeros(batch_size, 1)

    if use_gpu:
        print("use gpu for training")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        loss_func = loss_func.cuda()
        labels_one = labels_one.to("cuda")
        labels_zero = labels_zero.to("cuda")
    #训练
    loss_label = ['recons_loss', 'g_loss', 'd_loss','real_loss','fake_loss']
    animator = Animator(xlabel='ste', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=loss_label)
    start_time =time.time()
    for epoch in range(num_epochs):
        # 算每轮epoch的总体loss
        metric = Accumulator(len(loss_label))
        for i,(gt_images,labels) in enumerate(dataloader):
            #生成噪声z
            z =torch.randn(batch_size,latent_dim)
            if use_gpu:
                gt_images = gt_images.to("cuda")
                z = z.to("cuda")
            # 根据z生成器生成图像
            pred_images = generator(z)



            #优化判别器
            d_optimizer.zero_grad()
            # 判别器判别真实图片之后将概率结果放入损失函数并且优化生成器   pred_images.detach()是因为这时候pred_image不需要计算梯度，所以让它从计算图中分离出来
            real_loss = loss_func(discriminator(gt_images), copy.deepcopy(labels_one))
            fake_loss = loss_func(discriminator(pred_images.detach()), copy.deepcopy(labels_zero))
            d_loss = (real_loss + fake_loss)
            d_loss.backward()
            d_optimizer.step()

            # 优化生成器
            g_optimizer.zero_grad()
            # 适当引入重构loss，计算像素值的L1误差
            recons_loss = torch.abs(pred_images - gt_images).mean()
            # 判别器判别生成器生成的图片之后将概率结果放入损失函数并且优化生成器    这里的size 是Discriminator的(batch_size,1*28*28)
            g_loss = recons_loss * 0.05 + loss_func(discriminator(pred_images), copy.deepcopy(labels_one))
            g_loss.backward()
            g_optimizer.step()

            #需要和loss_label对应
            metric.add(recons_loss.item(),g_loss.item(),d_loss.item(),real_loss.item(),fake_loss.item())



            if i % 400 == 0:
                image = pred_images.data
                #save_image中的normalize设置成True，目的是将像素值min-max自动归一到【0,1】范围内，如果已经预测了【0,1】之间，则可以不用设置True
                torchvision.utils.save_image(image, f"{out_put_dir}/image_{len(dataloader)*epoch+i}_{name}.png", nrow=4,normalize=True)
        #保存模型
        torch.save(discriminator,f"{out_put_dir}/{model_dir}/discriminator_{name}.pth")
        torch.save(generator,f"{out_put_dir}/{model_dir}/generator_{name}.pth")
        current_time = time.time()
        print(
            f"running time:{int(current_time - start_time)}s,epoch:{epoch + 1},step:{len(dataloader) * (epoch+1) },",end="")
        loss_str ="".join([loss_label[index] + ':' + str(metric[index] / len(dataloader)) +"," for index in range(len(loss_label))])
        print(loss_str)
        animator.add(epoch + 1, [metric[index]/len(dataloader) for index in range(len(loss_label))])
        animator.save_png(f"{out_put_dir}/{loss_dir}/train_loss_epoch_{epoch+1}_{name}.png")

    animator.save(f"{out_put_dir}/{loss_dir}/train_loss_{name}.gif", f"{out_put_dir}/{loss_dir}/train_loss_{name}.png",fps=20)
train()
# 生成图像(预测)
model_generator =torch.load(f"{out_put_dir}/{model_dir}/generator_{name}.pth", weights_only=False)
model_discriminator=torch.load(f"{out_put_dir}/{model_dir}/discriminator_{name}.pth", weights_only=False)
fake_z = torch.normal(0,1,size=(batch_size,latent_dim))
print(fake_z.shape)
if use_gpu:
    print("use gpu for prediction")
    model_generator = model_generator.cuda()
    model_discriminator = model_discriminator.cuda()
    fake_z =fake_z.to("cuda")
pred_images_gener=model_generator(fake_z)
print(pred_images_gener.shape)
image = pred_images_gener.data
#save_image中的normalize设置成True，目的是将像素值min-max自动归一到【0,1】范围内，如果已经预测了【0,1】之间，则可以不用设置True
torchvision.utils.save_image(image, f"{out_put_dir}/{predict_dir}/pre_image_{name}.png", nrow=4,normalize=True)