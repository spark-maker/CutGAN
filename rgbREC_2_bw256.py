'''
连轮廓两边的点 连正方形的中心横线 保证脸（眼）一样宽，下巴额头发型都不管了 直方图均衡
'''
import cv2
import  numpy as np
import os
import dlib
from PIL import Image
predictor_path = ".\weight\dlibdat\shape_predictor_68_face_landmarks.dat"


file_path = ".\dataset\\mj"    # 输入文件夹
save_path1=".\dataset\\new1"    # 输入文件夹
if not os.path.exists(save_path1):
    os.makedirs(save_path1)


def grabcutw(img,ybegin=120,yend=370):

    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    bgdModle = np.zeros(SIZE, np.float64)
    fgdModle = np.zeros(SIZE, np.float64)
    # rect = ( 1,ybegin, img.shape[1], img.shape[0])
    rect = (1, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask2[i, j] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
    return img



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

top_size,bottom_size,left_size,right_size = (50,50,50,50)

for filename in os.listdir(file_path):   # 遍历输入路径，得到图片名
    imgpath = os.path.join(file_path, filename)
    img = cv2.imread(imgpath)

    height = img.shape[0]  # 例如：1181
    weight = img.shape[1]  # 例如：787
    ratio = height / weight

    img1 = cv2.resize(img, (200, int(200 * ratio)))  # 调整图像大小
    img1=grabcutw(img1)

    img2 = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))
    # 面部长方形
    img3 = img2.copy()

    dets = detector(img3, 1)
    for k, d in enumerate(dets):
        shape = predictor(img3, d)

    leb = shape.parts()[0]  # 左边缘
    # print(leb.x,leb.y)
    # cv2.circle(img3, (leb.x, leb.y), 4, (255, 255, 0), 2)

    reb = shape.parts()[16]  # 右边缘
    # cv2.circle(img3, (reb.x, reb.y), 4, (255, 0, 0), 2)

    # cv2.circle(img3, (jeb.x, jeb.y), 1, (255, 0, 255), 2)
    # print(reb.x, reb.y)
    weightH = int((reb.x - leb.x) * 1.0 / 2)
    cx = int((reb.x + leb.x) / 2)
    cy = int((reb.y + leb.y) / 2)
    # cv2.circle(img3, (cx,cy), 1, (0, 255, 255), 2) #中心点的位置显示

    # cv2.rectangle(img3, (cx-weightH,cy-weightH), (cx+weightH,cy+weightH), (255, 0, 255), 2)#根据三点画出的正方形
    weight = int(weightH * 2.2)
    # cv2.rectangle(img3, (cx-weight,cy-weight), (cx+weight,cy+weight), (255, 0, 255), 2)#裁剪示意

    # cv2.rectangle(img3, (cx-weightH,cy-weightH), (cx+weightH,cy+weightH), (255, 0, 255), 2)
    # cv2.imshow("canny3 ", img3)
    # cv2.waitKey()

    img4 = img3[cy - weight:cy + weight, cx - weight:cx + weight]
    # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)  # 彩色图转灰度
    img4 = cv2.resize(img4, (256, 256))
    # newimg = np.hstack((img4, img4))  # 裁剪好的照片
    # newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("canny3 ", img4)
    # cv2.waitKey()
    if filename.endswith(".jpeg"):
        filename1=filename[ :-5]
    else:
        filename1=filename[ :-4]
    filename=filename1+".jpg"
    cv2.imwrite(save_path1 + '/' + filename, img4)
    print(save_path1 + '/' + filename)
