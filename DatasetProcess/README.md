## README



#### (1)  /61 is mainly used to convert the VOC dataset downloaded from the Internet into COCO dataset.

- At present, it is only for single class. 

- VOC dataset should be like this:

  **--VOC dataset**

  ​      |--Annotations (*.xml)

  ​      |--images (*.png)

  converted COCO dataset will be like this:

  **--COCO dataset**

  ​      |--annotations

  ​            |--instances_test.json

  ​            |--instances_train.json

  ​            |--instances_val.json

  ​     |--train

  ​     |--test

  ​     |--val

  ​     |--xml (this may be useless, you can  delete it by yourself.)

- Possible data processing steps:

  - Rename images and XMLs at the same time.
  - Check XMLs <object>-<name>. ( There might be some mistakes for VOC dataset downloaded from Internet )
  - Check whether all images can be open correctly. (Some may have been broken.)
  - if everything is ok, go to `VocToCoco.py`.

####  (2) /others  is mainly used to transform the dataset annotated with labelme into COCO dataset.

​	Please refer to the following website: https://blog.csdn.net/weixin_42882838/article/details/102843082



