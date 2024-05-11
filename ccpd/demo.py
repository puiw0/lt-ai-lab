import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torch.autograd import Variable

from fh02 import fh02
from load_data import *

p = argparse.ArgumentParser()
p.add_argument('-i', '--input', type=str, required=True, help='path to the data folder(absolute path)')
p.add_argument('-m', '--model', type=str, required=False,
               help='path to the model file(absolute path). if not exists, please download from README')
args = p.parse_args()
data_path = args.input
model_path = args.model

use_gpu = torch.cuda.is_available()

img_size = (480, 480)
batch_size = 64 if use_gpu else 8
epochs = 25
saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
print("输出图片文件夹:", saving_dir)
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

num_class = 4
num_points = 4

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

if __name__ == '__main__':
    print("starting...")
    model = fh02(num_points, num_class)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # 使用已经训练好的模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 加载数据
    demo_data = demoTestDataLoader(data_path, img_size)
    # DataLoader = DataSet + Sampler
    demoDataLoader = DataLoader(demo_data, batch_size=1, shuffle=False, num_workers=1)

    start = datetime.now()
    for i, (XI, ims) in enumerate(demoDataLoader):

        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()

        img = cv2.imread(ims[0])
        left_up = [(cx - w / 2) * img.shape[1], (cy - h / 2) * img.shape[0]]
        right_down = [(cx + w / 2) * img.shape[1], (cy + h / 2) * img.shape[0]]
        cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)
        #   The first character is Chinese character, can not be printed normally, thus is omitted.
        lpn = alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + \
              ads[labelPred[6]]
        print('识别结果:', lpn)

        # PIL 图片上打印汉字
        pilImg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pilImg)
        font = ImageFont.truetype('simhei.ttf', size=40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text((int(left_up[0]), int(left_up[1]) - 40), lpn, (255, 0, 0),
                  font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        # PIL 图片转cv2 图片
        cv2charimg = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
        # cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1])-20), cv2.FONT_ITALIC, 2, (0, 0, 255))
        dstFileName = saving_dir + ims[0][-5:-4] + '.jpg'

        cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1]) - 20), cv2.FONT_ITALIC, 2, (0, 0, 255))
        cv2.imwrite(dstFileName, cv2charimg)
        print('图片保存地址:', dstFileName)

    end = datetime.now()
    print("用时: " + str(end - start))
