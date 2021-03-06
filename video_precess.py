import torch
if __name__ == '__main__':
    from utils.face_seg import FaceSeg
else:
    from .face_seg import FaceSeg
# from utils import *
import face_alignment
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DYLD_PRINT_LIBRARIES=1
import numpy as np
import time
from tqdm import tqdm
from math import *
from models import *
import pickle
import cv2

img_glo = None

# 整个图像纺射变换
# 图像裁剪
# 得到 纺射裁剪人脸

#### 还原
# 整个图像纺射变换
# 贴回原图

i = 0
cnt = 0
m_list = []
warp_shape = []
crop_list = []
cart_list = []
rgba_list = []

time1 = None
time2 = None

# 图像旋转
def rotate_bound(image, angle):    #https://www.jb51.net/article/144471.htm
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
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(0, 0, 0))


class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        # 缺少 人脸校准
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        # 仅检测人脸mask
        landmarks = self.__get_max_face_landmarks(image)
        # print('landmask = ', landmarks.shape)
        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)
            # return (image, landmarks, 0)
    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    # c = cos, s = sin
    # 旋转角度 redian=np.arctan
    def __rotate(image, landmarks):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))
        # append theta
        # image size after rotating
        height, width, _ = image.shape
        c = cos(radian)
        s = sin(radian)
        new_w = int(width * abs(c) + height * abs(s))
        new_h = int(width * abs(s) + height * abs(c))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[c, s, (1 - c) * width / 2. - s * height / 2. + Tx],
                      [-s, c, s * width / 2. + (1 - c) * height / 2. + Ty]])
        M2 = np.array([[c, -s, 0],
                      [s, c, 0]])
        M_inv = cv2.invertAffineTransform(M)

        m_list.append((M, M2, M_inv))
        warp_shape.append((new_w, new_h))


        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        # landmarks_rotate = landmarks
        return image_rotate, landmarks_rotate, radian




theta = 0


# 图像处理（人脸检测+ 校正+ 背景去除）
class Preprocess:
    def __init__(self, device='cpu', detector='dlib', mode='take'):

        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        self.alignment = ''
        self.segment = FaceSeg()
        self.mode = mode

    def process(self, image):
        global time1, time2
        # time1 = time.time()
        # tuple, (原图校正， 裁剪mask)
        face_info = self.detect.align(image)


        if face_info is None:
            return None

        # 人脸校正， mask校正。 校正的放大原图
        image_align, landmarks_align, theta = face_info

        # 人脸裁剪， 校正的放大人脸
        face = self.__crop(image_align, landmarks_align)
        # crop_shape = crop_list[cnt]

        mask = self.segment.get_mask(face)
        return np.dstack((face, mask))

    # @staticmethod
    def __crop(self, image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # cv2.imshow('img %d,%d,%d,%d' %(int(landmarks_top),int(landmarks_bottom) , int(landmarks_left),int(landmarks_right)), image[int(landmarks_top):int(landmarks_bottom) + 1, int(landmarks_left):int(landmarks_right) + 1])
        # cv2.waitKey()

        # expand bbox
        if self.mode == 'crop':

            top = int(landmarks_top - 0.4 * (landmarks_bottom - landmarks_top))
            bottom = int(landmarks_bottom + 0.1 * (landmarks_bottom - landmarks_top))
            left = int(landmarks_left - 0.1 * (landmarks_right - landmarks_left))
            right = int(landmarks_right + 0.1 * (landmarks_right - landmarks_left))
        else:
            top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
            bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
            left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
            right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        # 原代码
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255
        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        #存储真实的照片剪切位置
        crop_list.append((top, bottom, left, right))

        #返回的 face 来自于1920*1080的剪切
        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        cart_list.append((top_white,bottom_white+1, left_white,right_white+1))
        return image_crop

def rotate_bound2(image, angle):

    h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(0, 0, 0))


class Photo2Cartoon:
    def __init__(self, weight_path):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)

        params = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference_noseg(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face_rgba = cv2.cvtColor(face_rgba, cv2.COLOR_RGB2BGR)

        return face_rgba

    def inference(self, img):
        # face alignment and segmentation
        global cnt, i
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        print('[Step2: face detect] success!')
        # 保存 原 切图的大小
        size_temp = face_rgba.shape[:2]
        # mask_return = face_rgba[:, :, 3][:, :, np.newaxis].copy()
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        face_ = face.copy()

        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        background = (face_ * (1 - mask)).astype('uint8')
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

        face = (face * mask ) / 127.5 - 1
        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon, heatmap = self.net(face)[0][0], self.net(face)[2][0]


        # 显示热力图
        # heatmap = np.transpose((heatmap).numpy(), (1, 2, 0 ))

        # size = 256
        # x= heatmap
        # x = x - np.min(x)
        # cam_img = x / np.max(x)
        # cam_img = np.uint8(255 * cam_img)
        # cam_img = cv2.resize(cam_img, (size, size))
        # cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        # print(heatmap.shape)
        # cv2.imshow('img', cam_img)
        # cv2.waitKey()


        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        # cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = (cartoon * mask ).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        cartoon = cartoon + background


        # 图像旋转还原并保存
        M_ = m_list[cnt][0]
        M = m_list[cnt][2]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 还原cartoon 至 原切图大小, 并将原切图去白边 贴回原img大小中。
        # 条件还原
        cart = cv2.resize(cartoon, (size_temp), interpolation=cv2.INTER_AREA)

        # 首先对原图纺射变换
        img_ = cv2.warpAffine(img, M_, warp_shape[cnt], borderValue=(0, 0, 0))
        img_[crop_list[cnt][0]: crop_list[cnt][1] + 1, crop_list[cnt][2]: crop_list[cnt][3] + 1] = cart[cart_list[cnt][0]:cart_list[cnt][1],cart_list[cnt][2]:cart_list[cnt][3],:]
        img_ = cv2.warpAffine(img_, M, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0))
        # print('model process time: ', time2-time1)
        # cv2.imwrite(savepath[:-1]+'_result/' +args.mode+number+ "_frame%d.png" % i, img_)
        # cv2.imwrite(savepath + args.mode+number+ "_frame%d.png" % i, img)

        return cartoon, img_, img


def detect_while_CAM(model):

    global cnt, i
    cap = cv2.VideoCapture(0)
    while (True):
        # 逐帧捕获视频
        ret, frame = cap.read()
        # 对帧操作: 转成灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        (frame, img_, img) = model.inference(frame)

        cv2.imshow('frame', img_)
        if frame is None:
            ret, frame = cap.read()
            continue


        cnt += 1

        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()




# here
def video2ph(file_path, save_path, init_path, args):

    videoCapture = cv2.VideoCapture(file_path)
    success, frame = videoCapture.read()
    pre = Preprocess(mode=args.mode)
    global i, cnt
    timeF = 1
    while success:
        i = i + 1

        if (i % timeF == 0 ):
            frame = rotate_bound2(frame, 270)
            frame_ = frame.copy()
            # cv2.imshow('frame_', frame_)
            # cv2.waitKey()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 处理图像，返回人脸align_crop图像和mask

            face_rgba = pre.process(frame)


            if face_rgba is None:
                print('[Step2: face detect] can not detect face!!!')
                success, frame = videoCapture.read()
                continue
            rgba_list.append((face_rgba.shape[:2]))
            cv2.imwrite(init_path + args.f_name.split('.')[0][-4:] + "_init"+ str(i)+".jpg", frame_)
            # print('frame_ :', init_path + args.f_name.split('.')[0][-4:] + "_init"+ str(i)+".jpg")

            print('[Step2: face detect] success!')
            # 保存 原 切图的大小
            # size_temp = face_rgba.shape[:2]
            # mask_return = face_rgba[:, :, 3][:, :, np.newaxis].copy()
            face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
            face = face_rgba[:, :, :3].copy()

            mask = face_rgba[:, :, 3][:, :, np.newaxis].copy()
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path + args.f_name.split('.')[0][-4:] + "_frame%d.jpg" % i, face)
            cv2.imwrite(save_path + args.f_name.split('.')[0][-4:] + "_frame%d.png" % i, mask)

            cnt += 1
            print("frame%d.jpg" % i, ' saved.')
            print("-" * 50)

        success, frame = videoCapture.read()

    dict = {}
    dict['crop_list'] = crop_list
    dict['m_list'] = m_list
    dict['warp_shape'] = warp_shape
    dict['cart_list'] = cart_list
    dict['rgba_list'] = rgba_list
    dict['cnt'] = cnt

    with open('/Users/momo/Yugangtian/dict.pkl', "wb") as fp:
        pickle.dump(dict, fp)

def model_process(args, read_path, TmpPicSavePath):

    net = ResnetGenerator(ngf=32, img_size=256, light=True)
    params = torch.load(args.w, map_location='cpu')
    net.load_state_dict(params['genA2B'])
    print('[Step1: load weights] success!')



    dict_file = open('/Users/momo/Yugangtian/dict.pkl', 'rb')
    dict = pickle.load(dict_file)
    crop_list = dict['crop_list']
    m_list = dict['m_list']
    warp_shape = dict['warp_shape']
    cart_list = dict['cart_list']
    rgba_list = dict['rgba_list']
    max_cnt = dict['cnt']

    global i
    tmp = tqdm(range(1046, max_cnt))
    for cnt in tmp:
        i = i + 1
        tmp.set_description("model processing: %i" % i)
        try:

            face = cv2.imread(read_path + args.f_name.split('.')[0][-4:] + "_frame"+str(i)+".jpg")
            # cv2.imshow('face', face)
            # cv2.waitKey()

            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            mask = cv2.imread(read_path + args.f_name.split('.')[0][-4:] + "_frame"+str(i)+".png")
            mask = mask / 255.
            # print(mask)
            background = (face * (1 - mask)).astype('uint8')
            background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
            face = (face * mask) / 127.5 - 1


            face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
            face = torch.from_numpy(face)
            # print(face.shape)
            with torch.no_grad():
                cartoon, heatmap = net(face)[0][0], net(face)[2][0]


            cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
            cartoon = (cartoon + 1) * 127.5
            # cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
            cartoon = (cartoon * mask ).astype(np.uint8)
            cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
            cartoon = cartoon + background


            # 图像旋转还原并保存
            img = cv2.imread(TmpPicSavePath + args.f_name.split('.')[0][-4:] + "_init"+str(i)+".jpg" )
            # print(TmpPicSavePath + args.f_name.split('.')[0][-4:] + "_init"+str(i)+".jpg")

            M_ = m_list[cnt][0]
            M = m_list[cnt][2]
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 还原cartoon 至 原切图大小, 并将原切图去白边 贴回原img大小中。
            # 条件还原
            cart = cv2.resize(cartoon, rgba_list[cnt], interpolation=cv2.INTER_AREA)

            # 首先对原图纺射变换
            img_ = cv2.warpAffine(img, M_, warp_shape[cnt], borderValue=(0, 0, 0))
            img_[crop_list[cnt][0]: crop_list[cnt][1] + 1, crop_list[cnt][2]: crop_list[cnt][3] + 1] = cart[cart_list[cnt][0]:cart_list[cnt][1],cart_list[cnt][2]:cart_list[cnt][3],:]
            img_ = cv2.warpAffine(img_, M, (img.shape[1], img.shape[0]), borderValue=(0, 0, 0))
            cv2.imwrite(TmpPicSavePath+args.f_name.split('.')[0][-4:] + "_"+args.mode+str(i)+".jpg", img_)
            # print(TmpPicSavePath+args.f_name.split('.')[0][-4:] + "_"+args.mode+str(i)+".jpg")
        except:
            print("frame %d error."%i)
            pass



def ph2video(video_savepath, photo_path):
    dict_file = open('/Users/momo/Yugangtian/dict.pkl', 'rb')
    dict = pickle.load(dict_file)
    max_cnt = dict['cnt']
    videoWriter = cv2.VideoWriter(video_savepath, cv2.VideoWriter_fourcc(*'MJPG'), 25, (2160,1920))
    tmp = tqdm(range(1, max_cnt + 1))
    for i in tmp:
        # 加载图片，图片更多可以改变上面的10
        tmp.set_description("Video gen: %i" % i)
        try:
            # print(savepath+'frame' + str(i) + '.png')

            img_l = cv2.imread(photo_path+args.f_name.split('.')[0][-4:]+ "_init"+str(i)+".jpg")
            img = cv2.imread(photo_path+args.f_name.split('.')[0][-4:] + "_"+args.mode+str(i)+".jpg")
            img = np.concatenate((img_l, img), 1)

        except:
            print('Lack frame. pass...')
            continue

        # 如下让每张图显示1秒，具体与fps相等
        videoWriter.write(img)

    videoWriter.release()





if __name__ == '__main__':

    import argparse
    # 必须参数mode、 w、 f_nam。使用的测试视频默认存在file_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_name', type=str, default='IMG_5429.MOV')
    parser.add_argument('--src', type=str, default='/Users/momo/Yugangtian/test_video_phone/')
    parser.add_argument('--w', type=str, default='/Users/momo/Yugangtian/CycleGan_FakeSmile/photo2cartoon/models/_crop/photo2cartoon_crop_params_0310000.pt')
    parser.add_argument('--mode', type=str, default='crop')
    # 测试用参数
    parser.add_argument('--frame_start', type=int, default=0)

    args = parser.parse_args()

    yuantu_path = '/Users/momo/Yugangtian/test_video_ph/'
    file_path = '/Users/momo/Downloads/fakesmile_video/pre_video/'+args.f_name
    file_savepath = '/Users/momo/Downloads/fakesmile_video/post_video/'+args.f_name.split('.')[0]+args.mode+'_'+args.w.split('.')[0][-7:]+'.'+args.f_name.split('.')[-1]
    face_savepath = '/Users/momo/Yugangtian/face_detected/'
    TmpPicSavePath = '/Users/momo/Yugangtian/face2pic/'

    # video2ph(file_path, face_savepath, init_path=TmpPicSavePath, args=args)
    i = 0
    model_process(args, read_path=face_savepath, TmpPicSavePath=TmpPicSavePath)

    ph2video(video_savepath=file_savepath, photo_path=TmpPicSavePath)





# 摄像头视频测试
#     detect_while_CAM(c2p)
