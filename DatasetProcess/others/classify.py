import shutil
import cv2 as cv

sets=['train2019',  'val2019', 'test2019']
for image_set in sets:
    image_ids = open('./%s.txt'%(image_set)).read().strip().split()
    for image_id in image_ids:
        print(image_id)
        image_name = image_id[:-4]
        try:
            img = cv.imread('images/total2019/%s' % (image_id))
        except:
            img = cv.imread('images/total2019/%s' % (image_id))
        json='labelme/total2019/%s.json'% (image_name)
        cv.imwrite('images/%s/%s' % (image_set,image_id), img)
        cv.imwrite('labelme/%s/%s' % (image_set,image_id), img)
        shutil.copy(json,'labelme/%s/%s.json' % (image_set,image_name))
print("完成")

