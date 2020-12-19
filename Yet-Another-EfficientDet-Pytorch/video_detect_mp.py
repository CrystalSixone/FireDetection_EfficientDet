#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

# base_path = r'/rootfs/media/kasim/Data1/data/low_select_Person'
# out_path = r'/rootfs/media/kasim/Data1/data/low_select_Person'
base_path = r'/rootfs/media/kasim/DataSet/high_select_Person'
out_path = r'/rootfs/media/kasim/DataSet/high_select_Person'
# base_path = r'/rootfs/data/ErisedData/Record'
# out_path = r'/rootfs/data/ErisedData/Record'
video_list_file = r'file_list.txt'
gpu_count = 2


def parse_args():
    parser = argparse.ArgumentParser(description='Video Detect')
    parser.add_argument('--base_path', type=str, default=base_path, help='video base path')
    parser.add_argument('--list_file', type=str, default=video_list_file, help='video list file')
    parser.add_argument('--out_path', type=str, default=out_path, help='out path')
    parser.add_argument('--proccess_count', type=int, default=2, help='proccess count')
    parser.add_argument('--gpu_count', type=int, default=gpu_count, help='gpu count')
    args = parser.parse_args()
    return args


args = parse_args()

base_path = args.base_path
video_list_file = args.list_file
out_path = args.out_path
gpu_count = args.gpu_count

import sys
# sys.path.append('/opt/work/caffe/python')
sys.path.insert(0, '.')

import cv2
import numpy as np
import struct
import multiprocessing


############################################################################

SHOW = False
WRITE_FILE = True

SLEEP_TIME = 0
DEVICE = 'cuda'
BATCH_SIZE = 1

# SHOW = True
# THRESHOLDS = [
#     0.5,  # Person
#     0.7,  # Cat
#     0.7,  # Dog
#     0.5,  # BabyCar
#     0.5,  # Face
# ]

THRESHOLDS = [
    0.30,  # Person
    1.05,  # Cat
    1.05,  # Dog
    1.05,  # BabyCar
    1.05,  # Face
]

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
]


def get_box(img, result, score_thr=0.3, show=False, color=(0, 255, 0)):
    res = result[result[..., 4] >= score_thr]
    if res.size < 1:
        return None
    scores = res[..., -1]
    # bboxes = np.empty_like(res, dtype=np.int32)
    np.round(res[..., :-1], out=res[..., :-1])
    bboxes = res.astype(dtype=np.int32)
    height = img.shape[0]
    width = img.shape[1]
    np.clip(bboxes[..., 0], 0, width - 1, out=bboxes[..., 0])
    np.clip(bboxes[..., 2], 0, width - 1, out=bboxes[..., 2])
    np.clip(bboxes[..., 1], 0, height - 1, out=bboxes[..., 1])
    np.clip(bboxes[..., 3], 0, height - 1, out=bboxes[..., 3])

    bboxes = bboxes.tolist()
    scores = scores.tolist()
    for i, score in enumerate(scores):
        bboxes[i][-1] = score
    if show:
        for bbox in bboxes:
            label_text = '{:.3f}'.format(bbox[4])
            cv2.putText(img, label_text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, color)
            label_text = '{}'.format(bbox[2]-bbox[0])
            cv2.putText(img, label_text, (bbox[0], int(bbox[1]+32)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color)
            label_text = '{}'.format(bbox[3]-bbox[1])
            cv2.putText(img, label_text, (bbox[0], int(bbox[1]+64)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=1)
    return bboxes


def detect_proc(file_queue, out_queue, id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id % gpu_count)

    import torch
    import torch.onnx

    from torch.backends import cudnn
    from backbone import EfficientDetBackbone
    from efficientdet.utils import BBoxTransform, ClipBoxes
    from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

    compound_coef = 7
    force_input_size = None  # set None to use default size

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    def inference(model, images):

        results = []
        for image in images:
            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(image, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
            # result
            outs = invert_affine(framed_metas, out)
            # 只要person类别
            for out in outs:
                class_ids = out['class_ids']
                rois = out['rois']
                scores = out['scores']
                person_mask = class_ids == 0
                rois = rois[person_mask]
                if rois.size > 0:
                    scores = np.expand_dims(scores[person_mask], 1)
                    pred_box = np.concatenate([rois, scores], axis=1)
                    results.append(pred_box)
                else:
                    results.append(rois)
        return results

    count = 0
    cur_file_name = None
    try:
        # load model
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        model.load_state_dict(torch.load('weights/efficientdet-d{}.pth'.format(compound_coef)))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()
        win_name = 'process{}'.format(id)

        is_break = False
        while True:
            file_name = file_queue.get(timeout=5)
            if file_name is None:
                break
            cur_file_name = file_name
            img_path = os.path.join(base_path, file_name)
            cap = cv2.VideoCapture(img_path)
            # print('Proc: {}, {}'.format(id, img_path))

            if WRITE_FILE:
                out_data_file_name = os.path.splitext(img_path)[0]+'.ed.dat'
                out_dat_file = open(out_data_file_name, 'wb')

            imgs = []
            frame_id = 0
            bbox_count = 0
            while True:
                grabbed, image_bgr = cap.read()

                if SHOW:
                    if not grabbed:
                        break
                    imgs = [image_bgr]
                else:
                    if not grabbed:
                        if len(imgs) < 1:
                            break
                    else:
                        imgs.append(image_bgr)
                        if len(imgs) < BATCH_SIZE:
                            continue

                results = inference(model, imgs)

                for i, result in enumerate(results):
                    # print(i, imgs[i])
                    if result.size > 0:
                        j = 0
                        bboxes = get_box(imgs[i], result, score_thr=THRESHOLDS[j], show=SHOW, color=COLORS[j])
                        if (bboxes is not None):
                            if WRITE_FILE:
                                for k, bbox in enumerate(bboxes):
                                    # bbox_info = '{},{},{},{},{},{},{}\n'.format(frame_id, j, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])
                                    dat = struct.pack('6i1f', frame_id, j, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])
                                    # dat_list = struct.unpack('6i1f', dat)
                                    # print(len(dat), dat_list)
                                    out_dat_file.write(dat)
                                    bbox_count += 1
                    frame_id += 1
                    if SHOW:
                        cv2.imshow(win_name, imgs[i])

                imgs.clear()
                if SHOW:
                    if 0 == SLEEP_TIME:
                        k = cv2.waitKey()
                    else:
                        k = cv2.waitKey(SLEEP_TIME)
                    if k == 27:  # Esc key to stop
                        is_break = True
                        break
                else:
                    if not grabbed:
                        break

            if WRITE_FILE:
                out_dat_file.close()
                os.system('chmod a+wr {}'.format(out_data_file_name))
            out_queue.put((file_name, bbox_count, id))
            if is_break:
                break

            count += 1
            # if count % 10 == 0:
            #     print('Proc:', id, 'File Count:', count)
    except Exception as e:
        if str(e) != '':
            with open(os.path.join(out_path, 'error{}.txt'.format(id)), 'w') as file:
                error_info = 'Proc: {}, File Count: {}, Error File: {}, Error: {}\n'.format(id, count, cur_file_name, e)
                file.write(error_info)
                print(error_info)

    out_queue.put((cur_file_name, -1, id))

    if SHOW:
        cv2.destroyAllWindows()


def main():
    exclude_file_set = set()
    out_file_name = os.path.join(out_path, 'video_bbox_count.ed.txt')
    with open(out_file_name, 'r') as file:
        for line in file.readlines():
            exclude_file_set.add(line.split(',')[0].strip())

    video_list = []
    with open(os.path.join(base_path, video_list_file), 'r') as file:
        for video_name in file.readlines():
            video_name = video_name.split()[0].strip()
            if video_name in exclude_file_set:
                continue
            video_list.append(video_name)

    if len(video_list) <= 0:
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    workers = []
    for i in range(args.proccess_count):
        workers.append(multiprocessing.Process(target=detect_proc, args=(file_queue, out_queue, i,)))

    for i in range(args.proccess_count):
        workers[i].start()

    print('{:06f}, Start'.format(time.time()))

    if WRITE_FILE:
        out_file = open(out_file_name, 'a+')

    for video_path in video_list:
        full_video_path = os.path.join(base_path, video_path)
        if not os.path.exists(full_video_path):
            print(full_video_path, 'is not exists')
        file_queue.put(video_path)

    total_file_count = len(video_list)
    file_count = 0
    try:
        finish_worker_count = 0
        while True:
            file_info = out_queue.get(block=True)
            if file_info is None:
                break
            file_name, bbox_count, id = file_info
            if bbox_count < 0:
                print('Proc{} finish, last file: {}'.format(id, file_name))
                finish_worker_count += 1
                if args.proccess_count <= finish_worker_count:
                    break
                continue
            if WRITE_FILE:
                out_info = '{},{}\n'.format(file_name, bbox_count)
                out_file.write(out_info)
                out_file.flush()
                file_count += 1
                print('{:06f}, Proc{}, File Count: {}/{}, {}, {}'.format(time.time(), id, file_count, total_file_count, file_name, bbox_count))
                if file_count >= total_file_count:
                    break
    except Exception as e:
        print(e)

    for i in range(args.proccess_count):
        workers[i].join()

    if WRITE_FILE:
        out_file.close()
        os.system('chmod a+wr {}'.format(out_file_name))

    # for video_path in video_list:
    #     file_queue.put(video_path)
    #     detect_proc(file_queue, out_queue, 0)

    print('Finish!')


if __name__ == '__main__':
    main()
