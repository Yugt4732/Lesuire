import os
import cv2
from tqdm import tqdm
from math import *
import sys
import shutil
import time
from PIL import Image
import numpy as np
path = '/Users/momo/Yugangtian/all_naked_data/'
test_path =  '/Users/momo/Yugangtian/test/'
save_path = '/Users/momo/Yugangtian/final_naked_data/'
save_path2 = '/Users/momo/Yugangtian/train_naked/'
import warnings
warnings.filterwarnings('ignore')

 # 图像处理（取出mask部分）
def img_crop(path, save_path):
    for root, _, file_list in os.walk(path):
        # print(root,  file_list)
        for filename in tqdm(file_list):
            if 'jpg' in filename:
                im1 = Image.open(root+filename)
                im2 = Image.open(root+filename.split('.')[0]+'_seg.png')
                # im1.show()
                im1 = np.array(im1)
                im2 = np.array(im2)
                im2 = im2 / 256
                im = im1
                im[:, :, 0] = im1[:, :, 0] * im2
                im[:, :, 1] = im1[:, :, 1] * im2
                im[:, :, 2] = im1[:, :, 2] * im2
                im = Image.fromarray(im)
                # im.show()
                im.save(save_path+filename)

# 图像填充
def img_padding(path, save_path):

    for root, _, file_list in os.walk(path):
        # print(root,  file_list)
        for filename in tqdm(file_list):

            if filename.endswith('.jpg'):
                src = root + filename
                img = cv2.imread(src)
                shape = 1024
                high = img.shape[0]
                length = img.shape[1]

                top = int((shape - high) / 2)
                bottom = shape - high - top
                left = int((shape - length) / 2)
                right = shape - length - left

                newimg = cv2.copyMakeBorder(
                    img,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=[
                        0,
                        0,
                        0])
                cv2.imwrite(save_path+filename, newimg)


def get_mask_area(input_mask):
    ret, thresh = cv2.threshold(input_mask, 0, 255, cv2.THRESH_BINARY)
    ave = cv2.mean(thresh)[0] / 255
    area = ave * input_mask.shape[0] * input_mask.shape[1]
    return area

# 删除面积小于阙值的图像
def del_img(path, thread):
    cnt = 0
    for root, _, file_list in os.walk(path):
        # print(root,  file_list)
        for filename in tqdm(file_list):
            if filename.endswith('.jpg'):
                src = root + filename
                img = cv2.imread(src)
                img_matrix = np.array(img)
                area = get_mask_area(img_matrix)
                if area < thread:
                    os.remove(root+filename)
                    cnt += 1
    print(cnt)

#  图像左右切半
def img_crop2(path='/Users/momo/Downloads/samples'):
    save_path = path+'_croped'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.jpg'):
                src = os.path.join(root, filename)
                img = cv2.imread(src)
                w = img.shape[1]//2
                img_matrix = np.array(img)
                new_img = img_matrix[:,:w,:]
                cv2.imwrite(os.path.join(save_path, filename), new_img)

# 文件重命名，防止文件覆盖
def img_rename(path='/Users/momo/Yugangtian/trainA_croped'):
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.png'):
                print(filename)
                src = os.path.join(root, filename)
                os.rename(src, src.split('.')[0]+'_1.png')
            break
        break

# 文件删除
def img_del(path='/Users/momo/Downloads/output/'):
    for i in range(1,94223):
        try:
            os.remove('/Users/momo/Downloads/output/21_1_%d.jpg'% i)
            print(i)
        except:
            pass


# np.set_printoptions(threshold=np.inf)
def sub_mask(path, save_path):
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.jpg'):
                img_jpg = cv2.imread(os.path.join(root, filename))
                img_png = cv2.imread(os.path.join(root, filename.split('.')[0]+'_seg.png'), 0)

                cv2.imwrite(os.path.join(save_path, filename), img_jpg)

                # print(os.path.join(root, filename.split('.')[0]+'_seg.png'))
                # img_png = img_png.astype('uint8')
                # print(img_jpg.shape)
                # print(img_png)
                img_png = (img_png-127.5)
                img_png = np.clip(img_png, 0, 255)
                img_png = img_png*2
                img_png = np.clip(img_png, 0,255)
                img_jpg[:,:,0] =  img_jpg[:,:,0] * img_png/255
                img_jpg[:, :, 1] = img_jpg[:, :, 1] * img_png/255
                img_jpg[:, :, 2] = img_jpg[:, :, 2] * img_png/255
                cv2.imwrite(os.path.join(save_path, filename), img_jpg)

# 文件移动，可重命名。
def photo_move(path, save_path, start_num):
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.png'):
                start_num += 1
                shutil.copy(os.path.join(path, filename), os.path.join(save_path, str(start_num)+'.png'))

# 图像旋转功能实现
def rotate_bound2(image, angle):    #https://www.jb51.net/article/144471.htm
    '''
     . 旋转图片
     . @param image    opencv读取后的图像
     . @param angle    (逆)旋转角度
    '''

    h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))

# 图像旋转
def img_rotate(path, angle=0):
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(root, filename))
                rotated = rotate_bound2(img, angle)
                cv2.imwrite(os.path.join(root, filename), rotated)
                cv2.imshow("Rotated by %d Degrees" %angle ,rotated)
                time.sleep(5)
                # pass

def del_white(path, ):
    for root, _, file_list in os.walk(path):
        for filename in tqdm(file_list):
            if filename.endswith('.png'):
                img = cv2.imread(os.path.join(root, filename))
                for i in range(256):
                    for j in range(256):
                        if img[i, j, 0] >=240 and img[i, j, 1] >= 240 and img[i, j, 2] >= 240 :
                            img[i, j, 0] = 0
                            img[i, j, 1] = 0
                            img[i, j, 2] = 0
                cv2.imwrite(os.path.join(root, filename), img)

def floder_sub(path, sub_path, savepath):
    path_set = None
    subpath_set = None
    for root, _, path_file_list in  os.walk(path):

        print(len(path_file_list))
        path_set = set(path_file_list)
    for root, _, subpath_file_list in os.walk(sub_path):

        print(len(subpath_file_list))
        subpath_set = set(subpath_file_list)
    sub_set = path_set - subpath_set
    sub_set = list(sub_set)
    print(len(sub_set))

    for i in sub_set:
        # print(i)
        img = cv2.imread(os.path.join(path, i))
        cv2.imwrite(os.path.join(savepath, i), img)
        print(os.path.join(savepath, i))

if __name__ == '__main__':
    import torch

    src_path = '/Users/momo/Downloads/fake_init_/'
    dst_path = '/Users/momo/Yugangtian/photo2cartoon-master/dataset/photo2cartoon/trainA/'
    # for i in range(434, 1315):
    #     shutil.copy(os.path.join(dst_path, str(i) + '.png'), os.path.join(src_path, str(i) + '.png'))
    # photo_move(src_path, dst_path, 1023)

    path = '/Users/momo/Downloads/cart3d_init/'
    subpath = '/Users/momo/Downloads/cart3d/'
    save_path = '/Users/momo/Downloads/result/'
    floder_sub(path, subpath, save_path)