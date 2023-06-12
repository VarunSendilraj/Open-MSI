import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
"""
IMS_data = '20230528_caffeine_MSI_3.txt'
AbrationTime = 0.05054
"""
"""
IMS_data = '230317_caffeine_1.txt'
AbrationTime = 0.051435
"""

IMS_data = '230111_caffeini_3_195_1.txt'
#predefined time 
AbrationTime = 0.05054  #一点の照射時間


#open the file as read and split it by newline
with open(IMS_data, 'r') as file:
  list_1 = file.read().split("\n")

#preprocessing to covnert to csv
list_2 = []
for i in list_1:
  i = i.replace('\t',',')
  i = i.split(',')
  list_2.append(i)
list_2 = [[float(x) for x in y] for y in list_2]


#start time of the 
StartTime = 1.0  #照射開始時点
data_3 = []
list_3 = []
for i in list_2:
  if i[0] < StartTime:
    continue
  elif StartTime <= i[0] < StartTime + AbrationTime:
    data_3.append(i)
  else:
    data_3.append(i)
    list_3.append(data_3)
    StartTime += AbrationTime
    data_3 = []

Noise = 14 #バックグラウンドノイズ
list_4 = []
data_4 = 0
for i in list_3:
  for j in i:
    data_4 += max(j[1] - Noise,0)
  list_4.append(data_4)
  data_4 = 0

AbrationTimes = 10000 #ピクセル数
list__4 = [0] * AbrationTimes
for i in range(min(len(list__4), len(list_4))):
  if list_4[i] > 0:
    list__4[i] = list_4[i]
    list__4[i] = math.log(list__4[i], 10)

list_5 = []
length = 100 #画像の縦のピクセル数
for i in range(0, len(list__4), length):
  list_5.append(list__4[i:i + length])

list_6 = []
for i in range(len(list_5)):
  if i % 2 == 0:
    list_6.append(list_5[i])
  else:
    list_6.append(list_5[i][::-1])

image_1 = np.array(list_6)

plt.rcParams['font.size'] = 20
plt.imsave('MSI_image.bmp', image_1, cmap="jet")
plt.imshow(image_1, cmap="jet")
plt.colorbar(label='log(Signal intensity [arb. units])')

"""
plt.xticks([0, 33, 66, 100],[0, 1, 2, "3\nmm"], fontsize=20)
plt.yticks([0, 33, 66, 100],["3\nmm", 2, 1, 0], fontsize=20)
"""

plt.xticks([0, 25, 50, 75, 100],[0, 1, 2, 3, "4\nmm"], fontsize=20)
plt.yticks([0, 25, 50, 75, 100],["4\nmm", 3, 2, 1, 0], fontsize=20)

print(list_3)
print(list_4)
print(list_5)