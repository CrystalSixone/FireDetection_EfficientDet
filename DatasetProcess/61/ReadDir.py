# -*-coding:utf-8 -*-
# Author: w61
# Date: 2020.10.30

import sys,os
import glob
import xml.etree.ElementTree as ET
import json
import cv2

class ReadDir():
    def __init__(self,path):
        self.path = path
    
    def readXml(self,obj_name,overwrite=False):
        '''check whether <object>-<name> is right. 
        Args:
            obj_name: the name of the object
            overwrite: True or False
        '''
        xml_files = glob.glob(os.path.join(self.path,'*.xml'))
        names= []
        for i,item in enumerate(xml_files):
            tree = ET.parse(item)
            root = tree.getroot()
            for member in root.findall("object"):
                if member[0].text == obj_name:
                    member[0].text = obj_name
                    if overwrite:
                        tree.write(item)
                    print(item)
        for i,item in enumerate(xml_files):
            tree = ET.parse(item)
            root = tree.getroot()
            for member in root.findall("object"):
                if member[0].text not in names:
                    names.append(member[0].text)
        print(names)
    
    def readJson(self):
        img_paths = '/home/w61/FireDetection/Fire_dataset/COCO/train'
        #imgs = os.listdir(img_paths)
        # for img in imgs:
        #     src = os.path.join(img_paths,img)
        #     try:
        #         img_ = cv2.imread(src)
        #         img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #     except Exception as e:
        #         print(src)
        with open(self.path,'r')as f:
            data = json.load(f)

        # for cat in data['categories']:
        #     print(cat)
        for key in data.keys():
            print(key)
        count = 0
        for item in data["images"]:
            img_path = img_paths + "/" + item["file_name"]
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            except Exception as e:
                print(item["file_name"]) 
                count += 1
        print(count)
    
            
if __name__ == '__main__':
    xml_path = '/home/w61/PatternRecognition/smoke/Annotations/Annotations'
    json_path = '/home/w61/PatternRecognition/Yet-Another-EfficientDet-Pytorch/datasets/smoke/annotations/instances_val.json'
    json_path_2 = '/home/w61/FireDetection/Fire_dataset/COCO/annotations/instances_train.json'
    read = ReadDir(json_path_2)
    read.readJson()