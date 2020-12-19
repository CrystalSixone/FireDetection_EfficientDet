# -*-coding:utf-8 -*-
# Author: w61
# Date: 2020.10.27

'''
Description:
    The function of this module is to transform the data set in VOC format 
to COCO format. When instantiating class VocToCoco, you need to specify 
val_num, test_num, voc_xml_path, voc_img_path.
'''
import os
import random
import shutil
import sys
import json
import glob
import xml.etree.ElementTree as ET

class VocToCoco():
    def __init__(self,val_num,test_num,voc_xml_path,voc_img_path,coco_path=''):
        '''Initialize
        Args:
            val_num(int): the number of validation sets.
            test_num(int): the number of test sets.
            voc_xml_path(str): the path of VOC dataset storing Tags(.xml).
                            e.g. '/home/w61/PatternRecognition/Fire_dataset/VOC2020/Annotations'
            voc_img_path(str): the path of VOC dataset storing images.
                            e.g. '/home/w61/PatternRecognition/Fire_dataset/VOC2020/JPEGImages'
            coco_path(str)(Optional): the path to store the transformed COCO format dataset.
                            e.g. '/home/w61/PatternRecognition/Fire_dataset/COCO'
                            if not given, the path will be generated automatically.
        '''
        self.val_num = val_num
        self.test_num = test_num
        self.voc_xml_path = voc_xml_path
        self.voc_img_path = voc_img_path
        self.total_img = os.listdir(voc_img_path)
        self.total_num = len(self.total_img)
        self.train_num = self.total_num - self.val_num - self.test_num
        self.val_test_name = []
        self.START_BOUNDING_BOX_ID = 1
        self.PRE_DEFINE_CATEGORIES = None
        
        if coco_path == '':  
            split = self.voc_xml_path.split('/')
            del split[-2]
            del split[-1]
            for i,data in enumerate(split):
                if i == 0:
                    coco_path += data
                else:
                    coco_path += '/' + data
            coco_path += '/COCO'

        self.coco_path = coco_path 
    
    def mkdir(self,path):
        '''make directions.
        from https://www.php.cn/python-tutorials-424348.html
        '''
        path=path.strip()
        path=path.rstrip("\\")
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path+' ----- folder created')
            return True
        else:
            print(path+' ----- folder existed')
            return False
    
    ### Operation to XML
    def get(self,root,name):
        vars = root.findall(name)
        return vars
    
    def get_and_check(self,root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise ValueError("Can not find %s in %s." % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise ValueError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(vars))
            )
        if length == 1:
            vars = vars[0]
        return vars
    
    def get_filename_as_int(self,filename):
        try:
            filename = filename.replace("\\", "/")
            filename = os.path.splitext(os.path.basename(filename))[0]
            return int(filename)
        except:
            raise ValueError("Filename %s is supposed to be an integer." % (filename))

    def get_categories(self,xml_files):
        """Generate category name to id mapping from a list of xml files.

        Arguments:
            xml_files {list} -- A list of xml file paths.

        Returns:
            dict -- category name to id mapping.
        """
        classes_names = []
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        classes_names.sort()
        return {name: i for i, name in enumerate(classes_names)}

    ### Operations
    def makeCocoDir(self):
        '''make directions for COCO format.
        '''
        self.mkdir(self.coco_path)
        self.mkdir(self.coco_path + '/train')
        self.mkdir(self.coco_path + '/val')
        self.mkdir(self.coco_path + '/test')
        self.mkdir(self.coco_path + '/annotations')
        self.mkdir(self.coco_path + '/xml/xml_val')
        self.mkdir(self.coco_path + '/xml/xml_train')
        self.mkdir(self.coco_path + '/xml/xml_test')
    
    def generateTrainValTest(self):
        '''According to the number of requirements, 
        randomly generated sub dataset.
        '''
        # Val
        for i in range(self.val_num):
            random_img = random.choice(self.total_img)
            random_xml = random_img[:-4]+'.xml'
            self.val_test_name.append(random_img)
            source_img = os.path.join(self.voc_img_path,random_img)
            source_xml = os.path.join(self.voc_xml_path,random_xml)

            shutil.copy(source_img, self.coco_path + '/val')
            shutil.copy(source_xml, self.coco_path + '/xml/xml_val')
        
        # Test
        for i in range(self.test_num):
            random_img = random.choice(self.total_img)
            random_xml = random_img[:-4]+'.xml'
            self.val_test_name.append(random_img)
            source_img = os.path.join(self.voc_img_path,random_img)
            source_xml = os.path.join(self.voc_xml_path,random_xml)

            shutil.copy(source_img, self.coco_path + '/test')
            shutil.copy(source_xml, self.coco_path + '/xml/xml_test')
        
        # Train
        for img in self.total_img:
            if img in self.val_test_name:
                continue
            else:
                xml = img[:-4]+'.xml'
                source_img = os.path.join(self.voc_img_path,img)
                source_xml = os.path.join(self.voc_xml_path,xml)

                shutil.copy(source_img, self.coco_path + '/train')
                shutil.copy(source_xml, self.coco_path + '/xml/xml_train')
    
    def vocToCoco(self,xml_files,json_file):
        '''Transform VOC format to COCO format.
            main code below are from https://github.com/Tony607/voc2coco
        '''
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
        if self.PRE_DEFINE_CATEGORIES is not None:
            categories = self.PRE_DEFINE_CATEGORIES
        else:
            categories = self.get_categories(xml_files)
        bnd_id = self.START_BOUNDING_BOX_ID
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            path = self.get(root, "path")
            if len(path) == 1:
                filename = os.path.basename(path[0].text)
            elif len(path) == 0:
                print(111)
                filename = self.get_and_check(root, "filename", 1).text
            else:
                raise ValueError("%d paths found in %s" % (len(path), xml_file))
            ## The filename must be a number
            image_id = self.get_filename_as_int(filename)
            size = self.get_and_check(root, "size", 1)
            width = int(self.get_and_check(size, "width", 1).text)
            height = int(self.get_and_check(size, "height", 1).text)
            image = {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in self.get(root, "object"):
                category = self.get_and_check(obj, "name", 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category] + 1 ########## EfficientDet-Pytorch train started from 1.
                bndbox = self.get_and_check(obj, "bndbox", 1)
                xmin = int(self.get_and_check(bndbox, "xmin", 1).text) - 1
                ymin = int(self.get_and_check(bndbox, "ymin", 1).text) - 1
                xmax = int(self.get_and_check(bndbox, "xmax", 1).text)
                ymax = int(self.get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {"supercategory": "none", "id": cid, "name": cate}
            json_dict["categories"].append(cat)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        json_fp = open(json_file, "w")
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()

    def run(self):
        '''main function.
        '''
        self.makeCocoDir()
        self.generateTrainValTest()

        xml_val_files = glob.glob(os.path.join(self.coco_path, 'xml/xml_val', "*.xml"))
        xml_test_files = glob.glob(os.path.join(self.coco_path, 'xml/xml_test', "*.xml"))
        xml_train_files = glob.glob(os.path.join(self.coco_path, 'xml/xml_train', "*.xml"))

        self.vocToCoco(xml_val_files, self.coco_path + '/annotations/instances_val.json')
        self.vocToCoco(xml_test_files, self.coco_path + '/annotations/instances_test.json')
        self.vocToCoco(xml_train_files, self.coco_path + '/annotations/instances_train.json')
        
        print('Conver Finish. Go to check!')

if __name__ == '__main__':
    val_num = 40
    test_num = 0
    voc_xml_path = '/home/w61/FireDetection/Fire_dataset/fire1000_aug/new_ann'
    voc_img_path = '/home/w61/FireDetection/Fire_dataset/fire1000_aug/new_img'
    Trans = VocToCoco(val_num,test_num,voc_xml_path,voc_img_path)
    Trans.run()