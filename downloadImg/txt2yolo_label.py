import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test','val']
classes = ['football'] #需要更改
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('VOCData/Annotations/%s.xml' % (image_id)) #需要更改
    out_file = open('VOCData/labels/%s.txt' % (image_id), 'w') #需要更改
    tree = ET.parse(in_file)
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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('VOCData/labels/'):
        os.makedirs('VOCData/labels/')
    image_ids = open('VOCData/labels/%s.txt' % (image_set)).read().strip().split() #需要更改
    list_file = open('VOCData/%s.txt' % (image_set), 'w')
    print(image_set)
    for image_id in image_ids:
        list_file.write('VOCData/images/%s.jpeg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()


