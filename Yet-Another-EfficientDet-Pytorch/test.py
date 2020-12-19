"""
Simple Inference Script of EfficientDet-Pytorch
"""
import sys,os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import argparse

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, preprocess_batch, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

import glob
import yaml

# projects:
# final, fire, fire_300, fire_1000, smoke

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', required=True, help='project file that contains parameters')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('-i','--image',type=str, required=True, help='the path of image or the list path of images.')
    parser.add_argument('--is_saved',action='store_true',help='whether save the detection result.')
    parser.add_argument('--is_show',action='store_true',help='whether show the detection result.')
    parser.add_argument('--saved_path', type=str, default='result')

    args = parser.parse_args()
    return args

def display(preds, imgs, count=0, imshow=True, imwrite=True):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            print('{}:{}'.format(i,obj))
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite('./{}/{}_{}.jpg'.format(args.saved_path,count*30+i,'det'), imgs[i])

def list_of_groups(init_list, children_list_len):
    '''按数目对list分组
    '''
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

if __name__ == '__main__':
    args = get_args()
    compound_coef = 0
    force_input_size = None  # set None to use default size
    is_split_dir = False # 传入图片数量是否大于54张进行了list分割，默认为False
    is_dir = False # 传入是否为dir，默认为False

    img_path = args.image
    if os.path.isdir(img_path): # 如果传进来的是一个目录,则把该目录下所有图片作为输入
        is_dir = True
        img_path = glob.glob(os.path.join(img_path,'*'))
        if len(img_path) > 30: # 6G显存只能容纳一次输入54张图片
            img_path_batch = list_of_groups(img_path,20)
            is_split_dir = True    
    # print(img_path)
    
    project_path = 'projects/{}.yml'.format(args.project)
    project_params = Params(project_path)
    obj_list = project_params.obj_list

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                        ratios=anchor_ratios, scales=anchor_scales)
    # model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
    model.load_state_dict(torch.load('weights/weights_{}.pth'.format(args.project)))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    
    if not is_dir: # 传入的是单张图片
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        start_time = time.time() # 记录开始的时间
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        out = invert_affine(framed_metas, out)
        display(out, ori_imgs, imshow=args.is_show, imwrite=args.is_saved)
        fps = len(img_path) / (time.time()-start_time)
        print('----fps:{}----'.format(fps))
        
    
    elif not is_split_dir: # 没有进行图片分割
        ori_imgs, framed_imgs, framed_metas = preprocess_batch(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        start_time = time.time() # 记录开始的时间
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        out = invert_affine(framed_metas, out)
        display(out, ori_imgs, imshow=args.is_show, imwrite=args.is_saved)
        fps = len(img_path) / (time.time()-start_time)
        print('----fps:{}----'.format(fps))
        
    
    else: # 进行了图片分割
        inference_time = 0.0 # 记录每一分段的inference time
        for i,img_per_path in enumerate(img_path_batch):
            print('i:{}'.format(i))
            ori_imgs, framed_imgs, framed_metas = preprocess_batch(img_per_path, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            start_time = time.time() # 记录开始的时间
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)

                out = invert_affine(framed_metas, out)
                
                display(out, ori_imgs, count=i, imshow=args.is_show, imwrite=args.is_saved)
            
            inference_time += time.time() - start_time
        fps = len(img_path) / inference_time
        print('----fps:{}----'.format(fps))

    
