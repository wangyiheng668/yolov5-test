import xml.etree.ElementTree as ET  # 用于解析XML文件
import pickle  # 用于序列化数据
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile  # 用于复制文件

# classes=["ball","messi"]
classes=["ball"]  # 类别列表

TRAIN_RATIO = 80  # 定义训练比例


def clear_hidden_files(path):  # 递归清除指定目录下的隐藏文件
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)


def convert(size, box):  # 用于将边界框坐标从Pascal VOC格式转化到YOLO格式
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(image_id):  # 用于将边界框坐标从Pascal VOC格式转化到YOLO格式
    in_file = open('E:/code/pycham/yolo/datasets/VOCdevkit/VOC2007/Annotations/%s.xml' %image_id)
    out_file = open('E:/code/pycham/yolo/datasets/VOCdevkit/VOC2007/YOLOLabels/%s.txt' %image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


# VOCdevkit/
# └── VOC2007/
#     ├── Annotations/           # 存放Pascal VOC格式的标注文件
#     │   └── <image_id>.xml
#     ├── JPEGImages/            # 存放图像文件
#     │   └── <image_id>.jpg
#     ├── YOLOLabels/            # 存放YOLO格式的标签文件
#     │   └── <image_id>.txt
#     ├── images/                # 存放所有图像文件，包括训练集和验证集
#     │   ├── train/             # 训练集图像文件夹
#     │   └── val/               # 验证集图像文件夹
#     ├── labels/                # 存放所有标签文件，包括训练集和验证集
#     │   ├── train/             # 训练集标签文件夹
#     │   └── val/               # 验证集标签文件夹
#     ├── yolov5_train.txt       # 训练集图像路径文件
#     └── yolov5_val.txt         # 验证集图像路径文件
#  其中VOCdevkit->VOC2007->Annotations/JPEGImages是原有的

wd = os.getcwd()  # 获取当前工作目录
# base_dir = "E:\code\pycham\yolo\datasets"  # 指定希望创建的目标文件夹路径
# wd = r"E:\code\pycham\yolo\datasets"
data_base_dir = os.path.join(wd, "VOCdevkit/")  # VOC数据集的根目录
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)  # 不存在则创建
work_sapce_dir = os.path.join(data_base_dir, "VOC2007/")  # VOC2007数据集工作空间目录
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")  # 存放Pascal VOC格式的标注文件
if not os.path.isdir(annotation_dir):
        os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")  # 存放图像文件
if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")  # 存放YOLO格式的标签文件
if not os.path.isdir(yolo_labels_dir):
        os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov5_images_dir = os.path.join(data_base_dir, "images/")  # 存放所有图像文件
if not os.path.isdir(yolov5_images_dir):
        os.mkdir(yolov5_images_dir)
clear_hidden_files(yolov5_images_dir)
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")  # 存放所有标签文件
if not os.path.isdir(yolov5_labels_dir):
        os.mkdir(yolov5_labels_dir)
clear_hidden_files(yolov5_labels_dir)
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")  # 创建用于存放Train图像路径的文件yolov5_train.txt
if not os.path.isdir(yolov5_images_train_dir):
        os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_images_test_dir = os.path.join(yolov5_images_dir, "val/")  # 创建用于存放Val图像路径的文件yolov5_val.txt
if not os.path.isdir(yolov5_images_test_dir):
        os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
if not os.path.isdir(yolov5_labels_train_dir):
        os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)
yolov5_labels_test_dir = os.path.join(yolov5_labels_dir, "val/")
if not os.path.isdir(yolov5_labels_test_dir):
        os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)

train_file = open(os.path.join(work_sapce_dir, "yolov5_train.txt"), 'w', encoding='utf-8')
test_file = open(os.path.join(work_sapce_dir, "yolov5_val.txt"), 'w', encoding='utf-8')
train_file.close()
test_file.close()
train_file = open(os.path.join(work_sapce_dir, "yolov5_train.txt"), 'a', encoding='utf-8')
test_file = open(os.path.join(work_sapce_dir, "yolov5_val.txt"), 'a', encoding='utf-8')
list_imgs = os.listdir(image_dir) # list image files
prob = random.randint(1, 100)
print("Probability: %d" % prob)
# 通过遍历JPEGimages目录下的所有图像文件，将图像按照一定的比例划分成训练集和验证集
for i in range(0,len(list_imgs)):
    path = os.path.join(image_dir,list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)
    if(prob < TRAIN_RATIO): # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention) # convert label
            copyfile(image_path, yolov5_images_train_dir + voc_path)
            copyfile(label_path, yolov5_labels_train_dir + label_name)
    else: # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention) # convert label
            copyfile(image_path, yolov5_images_test_dir + voc_path)
            copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()
