# RealTimeMaskDetection
 
# 口罩辨識並使用OpenCV即時監測

基於深度學習的物件偵測(yolov3)，訓練出能辨認是否有戴口罩的資料集，使用Colab平台。

## 訓練結果

模型設定檔使用yolov3.cfg，訓練迭代2000次，辨識準確度平均約77.47%，但仍然有部分面部無法辨識。<br>

## 圖片辨識結果範例

![image](https://github.com/as147108/mask_detection/blob/main/Image/detection.PNG?raw=true)<br>
good: 96%　good: 88% good: 81%　good: 68%　bad: 89%　bad: 89%　bad: 79%

![image](https://github.com/as147108/mask_detection/blob/main/Image/detection3.PNG?raw=true)<br>
good: 99%　good: 66%　good: 62%　good: 54%　bad: 71%　bad: 69%　bad: 54%　none: 60%

資料來源:https://pjreddie.com/darknet/yolo/
