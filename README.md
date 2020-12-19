本博客使用到的EfficientDet开源代码：[ zylo117/Yet-Another-EfficientDet-Pytorch ](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## 数据库准备

### 1. 爬虫

1. 爬虫工具+浏览器：slenium+webdriver+chrome浏览器
2. 安装chrome浏览器：

```
	 1. wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
	 2. sudo dpkg -i google-chrome-stable_current_amd64.deb
```

3. 安装chrome-driver：

```
          1. LATEST=$(wget -q -O - http://chromedriver.storage.googleapis.com/LATEST_RELEASE)
          2.  wget http://chromedriver.storage.googleapis.com/$LATEST/chromedriver_linux64.zip
          3.  unzip chromedriver_linux64.zip
          4.  sudo mv -f chromedriver /usr/local/share/chromedriver
          5.  sudo ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver
          6.  sudo ln -sf /usr/local/share/chromedriver /usr/bin/chromedriver
```

4. 安装selenium：`pip install selenium`
5. 爬虫具体程序参见**test_baidu.py**

### 2. 数据清理+重命名

   - 删除cv2无法打开的图片：可使用`DatasetProcess/ImgRename.py def delWrongImg()`，但请注意此方法可能无法完全删除cv2打不开的图片，还需手动检查一遍。
   - 重命名图片：使用`DatasetProcess/ImgRename.py def rename()`

### 3. 使用labelImg标注

1. LabelImg的安装：https://github.com/tzutalin/labelImg （windows和Ubuntu都可以）
2. 打开labelImg：

```
cd labelImg # 下载labelImg的路径
python3 labelImg.py # 打开labelImg
```

3. 改变默认保存Annotations的路径: 点击左边`Change Save Dir`

4. 快捷键:

- W 新建矩形框
- Ctrl+S 保存当前标注
- A 前一张图
- D 后一张图

5. 若需要对标注后的图片和xml标注文件进行批量重命名，合并到同一文件夹/JPEGImage和/Annotations，请使用`DatasetProcess/ImgRename.py def folders_rename()`.

### 5.数据增强
参考链接：https://github.com/maozezhong/CV_ToolBox
具体代码详见`/CV_ToolBox/DataAugForObjectDetection/DataAugmentForObejctDetection.py`
### 4. 转化为COCO格式

1. 若使用LabelImg（.xml）进行图片标注，请参考`DatasetProcess/61/VocToCoco.py`
2. 若使用Labelme（.json）进行图片标注，请参考`DatasetProcess/others`

## 训练

（操作均在efficientDet目录下）

1. 将处理好的数据集存入efficientDet目录下的/datasets目录。
2. 在/projects目录下新建.yml文件，其中`project_name`需与datasets/中数据集文件夹名相同，`train_set`和`val_set`也需要与数据集目录下存放训练图片和验证图片的目录名称相同。`num_gpus`根据自己训练时需要的gpu进行设置。`obj_list`按照顺序填入数据集中包含的所有label名称。（可使用`DatasetProcess/61/ReadDir def readJson()`读取json文件中label的顺序）
3. 修改train.py中第303行，保存.pth的名称（避免和已有名称撞车）
4. 参考训练指令：`python train.py -c 2 -p fire2020 --batch_size 8 --lr 1e-5 --num_epochs 2 --load_weights weights/efficientdet-d2.pth`

## 测试

1. 查看logs：

```
cd logs/fire/tensorboard
tensorboard --logdir 20201027-105928 --host==127.0.0.1
```

2. test.py（可输入单张图片或整个图片文件夹目录）：`python test.py -p fire -i test --is_saved` 请注意，这里默认将保存的图片保存在result/目录下。
3. 测试一段视频：`python efficientdet_test_videos.py` 请进入代码修改视频路径，labels，和读取的pth文件路径。
4. 计算AP值：`python coco_eval.py -p fire -w weights/weights_fire.pth`
   **请注意，如果出现AP全为-1的情况，请注意修改coco_eval.py第147行的catIds**