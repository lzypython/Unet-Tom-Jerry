# <center>**基于迁移学习与并联注意力机制的猫和老鼠动画片语义分割**</center>

<font size=6>**目录:**</font>

<font size=4>**一、项目简介**</font>

<font size=4>**二、数据集处理**</font>

<font size=3>&emsp; **2.1、数据获取**</font>

<font size=3>&emsp; **2.2、数据清洗**</font>

<font size=3>&emsp; **2.3、数据标注**</font>

<font size=3>&emsp; **2.4、数据增强**</font>

<font size=3>&emsp; **2.5、数据集划分**</font>

<font size=4>**三、模型搭建**</font>

<font size=4>**四、模型训练**</font>


<font size=4>**五、模型改进**</font>

<font size=3>&emsp; **5.1 添加CBAM注意力机制**</font>

<font size=3>&emsp; **5.2 基于CBAM注意力机制的进一步改进——并联注意力机制**</font>

<font size=4>**六、模型对比**</font>

<font size=4>**七、项目总结**</font>

<font size=6 color=red>**亮点:**</font>

![image.png](attachment:368a0298-6c93-4491-9b51-dc0a9e720fc9.png)

# 一、项目简介

基于**Unet模型**和**并联注意力机制**实现猫和老鼠动画片的语义分割，其中并联注意力机制受**CBAM注意力机制**的启发，将**通道注意力机制机制和空间注意力机制并联**，给予不同通道不同空间不同权重。本项目使用的数据集通过**爬虫**获取，清洗后使用**百度平台EasyData**进行标注，共682张，其中545中用来训练，137张用来测试。本项目使用**Visio可视化模型结构图**，并对比了原版Unet、添加CBAM注意力机制和添加本项目提出的并联注意力机制三种模型。在项目的最后，通过进行模型测试，证明模型有较好的语义分割能力。

# 二、数据集处理

## 2.1 数据集获取

使用**爬虫**爬取百度图库中猫和老鼠图片，进行数据清晰和标注。脚本可以根据网页关键词进行爬取，并指定爬取的页数，保存在关键词同名目录下，如果文件夹不存在，那么会创建这个文件夹，图片名从0开始命名。本次项目是猫和老鼠的语义分割任务，因此**关键词指定为猫和老鼠，爬取页数指定为25页**，共爬取900张猫和老鼠的图片。


```python
# 导入相应的库
import os
import re
import requests
# 获取网站源码
def get_html(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    # 设置源代码的编码方式
    response.encoding = "utf-8"
    # return response.text
    if response.status_code == 200:
        return response.text
    else:
        print("网站源码获取错误")
def parse_pic_url(html):
    result = re.findall('thumbURL":"(.*?)"', html, re.S)
    return result
# 获取图片的二进制源码
def get_pic_content(url):
    response = requests.get(url)
    # 设置源代码的编码方式
    return response.content
# 保存图片
def save_pic(fold_name, content, pic_name):
    # with open("大熊猫/" + str(pic_name) + ".jpg", "wb") as f:
    with open(fold_name + "/" + str(pic_name) + ".jpg", "wb") as f:
        f.write(content)
        f.close()
# 定义一个新建文件夹程序
def create_fold(fold_name):
    # 加异常处理
    try:
        os.mkdir(fold_name)
    except:
        print("文件夹已存在")
# 定义main函数调用get_html函数
def get_image():
    # 输入文件夹的名字
    fold_name = input("请输入您要抓取的图片名字:")
    # 输入要抓取的图片页数
    page_num = input("请输入要抓取多少页？ (0. 1. 2. 3. .....)")
    # 调用函数，创建文件夹
    create_fold(fold_name)
    # 定义图片名字
    pic_name = 0
    # 构建循环，控制页面
    for i in range(int(page_num)):
        url = "https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&dyTabStr=MCwzLDEsMiw2LDUsNCw4LDcsOQ%3D%3D&word=%E7%8C%AB%E5%92%8C%E8%80%81%E9%BC%A0"
        headers = {
            "Accept": "text/plain, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive",
            "Cookie": "BDqhfp=%E5%A4%A7%E7%86%8A%E7%8C%AB%E5%9B%BE%E7%89%87%26%26NaN-1undefined%26%261632%26%263; BIDUPSID=D076CA87E4CD25BA082EA0E9B5B9C82F; PSTM=1663428044; MAWEBCUID=web_fMcFGAgtkEbzDpinjKvUtGFDInsruypyhIDrXDSpxBBJoXftlZ; BAIDUID=D076CA87E4CD25BA568D2D9EF1AD5F5C:SL=0:NR=10:FG=1; indexPageSugList=%5B%22%E7%8C%AB%22%2C%22%26cl%3D2%26lm%3D-1%26ie%3Dutf-8%26oe%3Dutf-8%26adpicid%3D%26st%3D%26z%3D%26ic%3D%26hd%3D%26latest%3D%26copyright%3D%26word%3D%E5%A4%A7%E8%B1%A1%26s%3D%26se%3D%26tab%3D%26width%3D%26height%3D%26face%3D%26istype%3D%26qc%3D%26nc%3D%26fr%3D%26expermode%3D%26force%3D%26pn%3D30%26rn%3D30%22%2C%22%E6%80%A7%E6%84%9F%E7%BE%8E%E5%A5%B3%22%5D; ZFY=JujkjWiLPjOsSz:Ag1v0hFWlSBt4qjPC4L6bB4MDS6Jo:C; BAIDUID_BFESS=D076CA87E4CD25BA568D2D9EF1AD5F5C:SL=0:NR=10:FG=1; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=null; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; ab_sr=1.0.1_YTc4N2NiNWIyZWM5NTkzYzQ3MmZlNTI3Y2YyM2RiMTE3YmYwMTBiNzQ0YzhlZmJkZDY4YjJhZWU4NjVmMmQxZmJkYTcxODZkYTgwNjhhZDY5ZWZmYjg4Y2FmMGE5YTBmNjc3M2JhZDEwZTU1MTAyMTA1MjUxN2Y2NDNlMTJiNzhjNTIyYTQwNTg5ODNiMzc1MjRlZDdmNTVkMzdkOGJiOQ==",
            "Host": "image.baidu.com",
            "Referer": "https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word=%B4%F3%D0%DC%C3%A8%CD%BC%C6%AC&fr=ala&ala=1&alatpl=normal&pos=0&dyTabStr=MTEsMCwxLDMsNiw1LDQsMiw3LDgsOQ%3D%3D",
            "Sec-Ch-Ua": '"Microsoft Edge";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.43",
            "X-Requested-With": "XMLHttpRequest",

        }
        params = {
            "tn": "resultjson_com",
            "logid": "11637882045647848541",
            "ipn": "rj",
            "ct": "201326592",
            "fp": "result",
            "fr": "ala",
            "word": fold_name,
            "queryWord": fold_name,
            "cl": "2",
            "lm": "-1",
            "ie": "utf-8",
            "oe": "utf-8",
            "pn": str(int(i + 1) * 30),
            "rn": "30",
            "gsm": "3c",
        }
        html = get_html(url, headers, params)
        # print(html)
        result = parse_pic_url(html)

        # 使用for循环遍历列表
        for item in result:
            # print(item)
            # 调用函数，获取图片的二进制源码
            pic_content = get_pic_content(item)
            # 调用函数保存图片
            save_pic(fold_name, pic_content, pic_name)
            pic_name += 1
            # print(pic_content) # 二进制源码
            print("正在保存" + str(pic_name) + " 张图片")
get_image()
```

## 2.2数据清洗

由于数据是爬取的，**质量参差不齐**，尤其是分辨率不统一，后续输入模型中的图像要求大小统一，所以要resize，为防止resize对图片造成的失真，这里对**图像的长宽比**进行判断，如果长宽比小于0.5或者大于2，那么就舍弃这张图片，如果在范围之类那么resize成**512x512**大小的图片。并且对通过代码筛选的**图片内容**进行**人工逐一筛查**，如图片中没有出现猫或者老鼠的图片，或者清晰度特别低的脏数据，进行剔除。如下面所示左侧为长宽比不符合要求，右侧为图片内容不符合要求，故都进行剔除！

![image.png](attachment:1e8df6d6-d879-4d73-adeb-3ba9620bd535.png)
![484.jpg](attachment:b780ea95-74e5-417f-b2de-92c225c28d02.jpg)


```python
import shutil
import os
from PIL import Image
# Create a directory for the resized images if it doesn't exist
result_directory = "result"
directory_path = "猫和老鼠"
os.makedirs(result_directory, exist_ok=True)

# Initialize a counter for naming the resized images
image_counter = 0

# Iterate through each file in the directory again
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            with Image.open(os.path.join(directory_path, filename)) as img:
                # Calculate the aspect ratio
                aspect_ratio = img.width / img.height

                # Resize and save the image if it meets the aspect ratio criteria
                if 0.5 < aspect_ratio < 2:
                    # Resize the image
                    resized_img = img.resize((512, 512))
                    # Define the new filename
                    new_filename = f"{image_counter}.png"
                    new_filepath = os.path.join(result_directory, new_filename)

                    # Save the resized image
                    resized_img.save(new_filepath)

                    # Increment the counter
                    image_counter += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
print(image_counter)
```

![image.png](attachment:71027af1-d3c7-4976-8a21-716bee16d22c.png)

最终获得682张图片，统一为512x512大小，内容都符合要求。

## 2.3数据标注

刚开始时使用**labelme**进行标注，如左图所示，通过点连线确定边缘。但是标注了几十个发现标注速度过于慢，且边缘不够清晰。于是转用百度的**EasyData**进行标注，如右图所示，通过点击主体进行区域选择，左键绿点表示添加区域，右键红点表示减少区域，效率提高了不少。**最终标注682张数据，用时8个小时**。
![image.png](attachment:1cf9e34c-d78a-4f9d-b9af-52f3fb8e78ce.png)



![image.png](attachment:16840e93-6486-424b-8697-d00098435fca.png)

## 2.4 数据增强

为了提高模型泛化能力，帮助模型更好地适应各种复杂的现实场景和变化，减少过拟合的风险，图片读入到网络前还会通过随机缩放、翻转、色域变换进行数据增强。


```python
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".png"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        #   翻转图像
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label
        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #   应用变换
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data, label

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels

```

## 2.5 数据集划分

共获得标注数据682张，按8:2的比例划分训练集和测试集，共545张数据集用于训练,137张数据用于测试，如图所示：

![image.png](attachment:2ef70eb5-6775-4f0e-bfb7-d3fae374bb31.png)

# 三、模型搭建

本项目语义分割基于**Unet**模型实现，其中编码器使用**Resnet50**，预训练权重使用在VOC数据集上训练的Resnet50，解码器基于**双线性插值**实现。使用**Visio**画出模型结构图如下：

![image.png](attachment:772eb3ae-d8fc-4997-9e25-0a2bf5dee82e.png)

通过torch.load加载预训练模型权重：


```python
if model_path != '':
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    if local_rank == 0:
        print('Load weights {}.'.format(model_path))

    #------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    #------------------------------------------------------#
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    #------------------------------------------------------#
    #   显示没有匹配上的Key
    #------------------------------------------------------#
    if local_rank == 0:
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
```

编码器部分代码如下：


```python
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from nets.Myattention import cbam_block 
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------------------------------#
        #   输入图像为512,512,3
        #   当我们使用resnet50的时候
        #-----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 512,512,3 -> 256,256,64
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        # 245,256,64 -> 128,128,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        # 128,128,64 -> 128,128,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 128,128,256 -> 64,64,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 64,64,512 -> 32,32,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 32,32,1024 -> 16,16,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)
        in_channel = feat1.size()[1]
        cbam1= cbam_block(in_channel)
        feat1 = cbam1(feat1)
        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)
        feat3   = self.layer2(feat2)
        feat4   = self.layer3(feat3)
        feat5   = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'), strict=False)
    del model.avgpool
    del model.fc
    return model

```

解码器代码如下，其中上采样采用双线性插值完成：


```python
import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        # upampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

```

# 四、模型训练

设置模型所需超参数，训练过程采用**参数冻结**形式训练。开始训练时冻结编码器的参数，可以减少需要更新的参数数量，从而加快训练过程的速度。后进行解冻，更新全部参数。学习率采用**余弦退火**方式递减，初始时较大的学习率可以加快模型的收敛速度，但随着训练的进行，逐渐减小学习率可以使模型更加稳定地收敛到最优解。


```python
num_classes=3
Init_lr=0.001
Freeze_Epoch =4
UnFreeze_Epoch=5
Freeze_batch_size=16
Unfreeze_batch_size=8
```

开始训练，为后续**模型可视化对比**，在训练过程中以日志形式保存模型loss损失


```python
from train import train # train为自定义文件
train(num_classes,Init_lr,Freeze_Epoch,UnFreeze_Epoch,Freeze_batch_size,Unfreeze_batch_size)
```

模型曲线如下所示，可以看到设置的冻结轮数为10 ，在第10轮的时候，训练曲线有明显的下降趋势，后续慢慢收敛

![epoch_loss.png](attachment:64366658-978a-4a73-aa63-87fbcc9102dd.png)

为评估所训练模型的性能，采用**Iou**和**F1值**进行评测，IOU和F1值在语义分割任务中能够说明模型对每个类别的像素级别预测的准确性和覆盖程度，是评估模型分割性能的重要指标。


```python
import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

def get_miou():
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 3
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","Tom","Jerry"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
get_miou()
```

![image.png](attachment:2a361c85-824b-4fc9-9477-8bb5319b524e.png)
![image.png](attachment:876a1201-3c05-48d5-8f14-810413f2fc4a.png)

模型预测

# 五、模型改进

### 5.1 添加CBAM注意力机制

CBAM注意力机制是一种用于增强卷积神经网络性能的注意力模块。它可以帮助网络在学习特征的过程中更好地关注重要的区域，从而提高模型的性能和泛化能力。

CBAM注意力机制包括两个关键部分：通道注意力模块和空间注意力模块。通道注意力模块用于学习特征图中不同通道之间的相关性，从而提高网络对不同特征的关注度。而空间注意力模块则用于学习特征图中不同空间位置之间的相关性，从而提高网络对不同位置的关注度。

![image.png](attachment:7f9a3d55-cd2e-4454-bcc4-ce513439f622.png)
![image.png](attachment:53e3566e-7040-4dee-9494-b7d7e27f3fb2.png)
![image.png](attachment:af00ed1b-dbc3-4d7d-accf-3a2c1ed19daa.png)


```python
# 代码实现
import torch
import torch.nn as nn
import math
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.max_pool = nn.AdaptiveMaxPool2d(1).cuda()

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False).cuda()

        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

```

将CBAM注意力机制添加到编码器中，使编码器每一个特征图都经过CBAM注意力机制处理，用Visio画出模型结构图，其中紫色模块代表CBAM注意力机制模块：
![image.png](attachment:99bdb7c2-6280-41e7-aea1-6e7b3c63965d.png)


```python
import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------------------------------#
        #   输入图像为512,512,3
        #   当我们使用resnet50的时候
        #-----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 512,512,3 -> 256,256,64
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        # 245,256,64 -> 128,128,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        # 128,128,64 -> 128,128,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 128,128,256 -> 64,64,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 64,64,512 -> 32,32,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 32,32,1024 -> 16,16,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)
        in_channel1 = feat1.size()[1]
        cbam1= cbam_block(in_channel1)
        feat1 = cbam1(feat1)
        
        
        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)
        in_channel2 = feat2.size()[1]
        cbam2= cbam_block(in_channel2)
        feat2   = cbam2(feat2)
        
        feat3   = self.layer2(feat2)
        in_channel3 = feat3.size()[1]
        cbam3= cbam_block(in_channel3)
        feat3   = cbam3(feat3)
        
        feat4   = self.layer3(feat3)
        in_channel4 = feat4.size()[1]
        cbam4= cbam_block(in_channel4)
        feat4   = cbam4(feat4)
        
        feat5   = self.layer4(feat4)
        in_channel5 = feat5.size()[1]
        cbam5= cbam_block(in_channel5)
        feat5   = cbam5(feat5)
        return [feat1, feat2, feat3, feat4, feat5]
```

训练改进后添加CBAM注意力机制的模型


```python
from train import train # train为自定义文件
train(num_classes,Init_lr,Freeze_Epoch,UnFreeze_Epoch,Freeze_batch_size,Unfreeze_batch_size)
```

进行IOU和F1评估


```python
get_miou()
```

    Load model.
    logs/best_epoch_weights.pth model, and classes loaded.
    Configurations:
    ----------------------------------------------------------------------
    |                     keys |                                   values|
    ----------------------------------------------------------------------
    |               model_path |              logs/best_epoch_weights.pth|
    |              num_classes |                                        3|
    |                 backbone |                                 resnet50|
    |              input_shape |                               [512, 512]|
    |                 mix_type |                                        1|
    |                     cuda |                                     True|
    ----------------------------------------------------------------------
    Load model done.
    Get predict result.


    100%|██████████| 137/137 [00:09<00:00, 13.85it/s]


    Get predict result done.
    Get miou.
    Num classes 3
    ===>background:	Iou-93.27; Recall (equal to the PA)-97.1; Precision-95.94; F1-96.52
    ===>Tom:	Iou-69.1; Recall (equal to the PA)-77.35; Precision-86.62; F1-81.72
    ===>Jerry:	Iou-55.91; Recall (equal to the PA)-73.79; Precision-69.76; F1-71.72
    ===> mIoU: 72.76; mPA: 82.75; Accuracy: 93.62
    Get miou done.
    Save mIoU out to miou_out/mIoU.png
    Save mPA out to miou_out/mPA.png
    Save Recall out to miou_out/Recall.png
    Save Precision out to miou_out/Precision.png
    Save confusion_matrix out to miou_out/confusion_matrix.csv


![image.png](attachment:426557d0-2bce-4369-ad0d-10742c8ad3b5.png)

添加CBAM注意力机制后，各类别F1和Iou均有提高，说明CBAM的有效性，尤其是目标体较小的Jerry有较明显的提升。

### 5.2 基于CBAM注意力机制的进一步改进——并联注意力机制

受CBAM注意力机制启发，本项目通过并联通道注意力机制和空间注意力机制，帮助网络更好地捕捉图像中的重要信息。使用Visio画出并联注意力机制的示意图：

![image.png](attachment:2bdc4c16-f9f5-4440-a9f4-1fc5d42493ce.png)
![image.png](attachment:9678f11c-7702-4727-b52c-801ebc6e18b8.png)
![image.png](attachment:646d98b2-0963-435b-a7ba-a0b55147f5ae.png)


```python
# 代码实现
import torch
import torch.nn as nn
import math
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.max_pool = nn.AdaptiveMaxPool2d(1).cuda()

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False).cuda()

        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class parallel_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(parallel_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        CA = x * self.channelattention(x)
        SA = x * self.spatialattention(x)
        out = CA+SA 
        return out
```

通过Visio画出模型结构图，图中模块P代表并联注意力机制：
![image.png](attachment:a488eaae-dea8-4b71-bfb1-d76d99f0de3e.png)

训练添加并联注意力机制的模型


```python
from train import train # train为自定义文件
train(num_classes,Init_lr,Freeze_Epoch,UnFreeze_Epoch,Freeze_batch_size,Unfreeze_batch_size)
```

评估改进后并联注意机制模型的Iou和F1


```python
get_miou()
```

    Load model.
    logs/best_epoch_weights.pth model, and classes loaded.
    Configurations:
    ----------------------------------------------------------------------
    |                     keys |                                   values|
    ----------------------------------------------------------------------
    |               model_path |              logs/best_epoch_weights.pth|
    |              num_classes |                                        3|
    |                 backbone |                                 resnet50|
    |              input_shape |                               [512, 512]|
    |                 mix_type |                                        1|
    |                     cuda |                                     True|
    ----------------------------------------------------------------------
    Load model done.
    Get predict result.


    100%|██████████| 137/137 [00:10<00:00, 13.37it/s]


    Get predict result done.
    Get miou.
    Num classes 3
    ===>background:	Iou-93.99; Recall (equal to the PA)-97.75; Precision-96.07; F1-96.9
    ===>Tom:	Iou-70.89; Recall (equal to the PA)-79.16; Precision-87.16; F1-82.97
    ===>Jerry:	Iou-61.16; Recall (equal to the PA)-73.07; Precision-78.95; F1-75.9
    ===> mIoU: 75.35; mPA: 83.33; Accuracy: 94.34
    Get miou done.
    Save mIoU out to miou_out/mIoU.png
    Save mPA out to miou_out/mPA.png
    Save Recall out to miou_out/Recall.png
    Save Precision out to miou_out/Precision.png
    Save confusion_matrix out to miou_out/confusion_matrix.csv


# 六、模型对比

将上述三种模型进行对比。分别为**原版Unet、添加CBAM注意力机制的Unet和添加并联注意力机制的Unet**，评价指标为各类的**Precision、Recall、F1、MeanF1、MeanIou、Accuracy**

![image.png](attachment:b448f38c-329b-4ff2-bf09-c56ef1ffc050.png)

![image.png](attachment:caea0ea4-b755-4c05-b7d1-b2861207d4db.png)

从评价指标拍数据来看，本项目提出的并联注意力机制在各项指标上都达到了**最优指标**，说明创新点的有效性。

对比Train loss和Val loss曲线可以看出，使用本项目提出的并联注意力机制loss收敛速度更快，且收敛loss的损失要低于其他两组。

![image.png](attachment:2fb21735-02b3-4b98-8dc2-cc096ebaf76f.png)
![image.png](attachment:486ebc98-6120-4dbc-bd7a-c5c4d6262563.png)

对比不同模型的预测效果图可以看出，并联注意力机制效果图整体上更清晰，在细节的处理上优于其他对比模型。由于猫和老鼠中的老鼠属于小物体，其细节更难捕捉，如**老鼠的尾巴**，对比不同的预测效果图可以看出并联注意力机制下老鼠的尾巴更加完整，更接近于真实标签，说明提出的并联注意力机制的优越性。

![image.png](attachment:92a7baf7-0011-4b1c-b66e-cf446c80f137.png)

# 七、项目总结

本项目使用**爬虫技术**获取数据集，经过**数据清洗**后使用**百度EasyData**平台实现数据标注，通过**Pytorch框架**搭建**Unet模型**，使用**Resnet50**作为编码器。使用**迁移学习**的思想，加载Resnet50在VOC数据集上训练好的**预训练模型**，从而减少计算量和资源的消耗。通过引入**CBAM注意力机制**，加强模型对图像中重要信息的捕捉。在此基础上，继续改进，提出了**并联注意力机制**，通过并联通道注意力机制和空间注意力机制进一步加强模型的对不同信息的关注度。通过实验证明，所提出的**并联注意力机制的有效性**。

通过本学期认真的学习和这次完整的大作业，让我对神经网络有了全新的认识，从之前之后使用别人搭建好的模型，到现在自己也能够修改模型结构，提出一些小的创新点，并通过实验证明所提创新点的有效性，还是很有成就感的。

<font size=4>**最后，感谢刘老师这一个学期的付出，您辛苦了！**</font>
