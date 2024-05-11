# CCPD车牌识别项目



## 运行
下载作者已训练好的模型
1. wR2 模型
2. fh02 模型

```shell
python demo.py -m <absolute path of model fh02 file> -i <absolute path of demo img dir>
```

## 数据集

CCPD数据集没有专门的标注文件，每张图像的文件名就是该图像对应的数据标注。例如图片3061158854166666665-97_100-159&434_586&578-558&578_173&523_159&434_586&474-0_0_3_24_33_32_28_30-64-233.jpg 的文件名可以由分割符’-'分为多个部分：

1. 3061158854166666665为区域（这个值可能有问题，可以不管）；
2. 97_100对应车牌的两个倾斜角度-水平倾斜角和垂直倾斜角, 水平倾斜97度, 竖直倾斜100度。水平倾斜度是车牌与水平线之间的夹角。二维旋转后，垂直倾斜角为车牌左边界线与水平线的夹角。CCPD数据集中这个参数标注可能不那么准，这个指标具体参考了论文Hough Transform and Its Application in Vehicle License Plate Tilt Correction；
3. 159&434_586&578对应边界框左上角和右下角坐标:左上(159, 434), 右下(586, 578)；
4. 558&578_173&523_159&434_586&474对应车牌四个顶点坐标(右下角开始顺时针排列)：右下(558, 578)，左下(173, 523)，左上(159, 434)，右上(586, 474)；
5. 0_0_3_24_33_32_28_30为车牌号码（第一位为省份缩写），在CCPD2019中这个参数为7位，CCPD2020中为8位，有对应的关系表；
6. 64为亮度，数值越大车牌越亮（可能不准确，仅供参考）；
7. 233为模糊度，数值越小车牌越模糊（可能不准确，仅供参考）。