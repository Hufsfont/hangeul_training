import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 

#jupyter 노트북 환경에서 동작
#reference : youtube @sentdex
#이미지 위치나 카테고리는 임의로 설정. 이미지 데이터가 아직 없으므로

DATADIR = "C:/Images" # 이미지 위치
CATEGORIES = ["ㄱ","ㄴ","ㄷ"] #이미지 카테고리

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #아까 설정해 둔 이미지 위치
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
        #이미지를 불러와서 그레이스케일화 - 2차원 행렬로 나타내기 위해 / 그 후 행렬 형태로 저장
        plt.imshow(img_array, cmap="gray") #이미지가 그레이스케일화 되었으므로 색깔 맵은 그레이
        plt.show() #이미지 보여주기
        break
    break #여기까지 하면 그레이스케일 처리된 사진 이미지가 출력됨

IMG_SIZE = 50 #이미지를 리사이즈하기위해 새로 정할 이미지 사이즈를 초기화
#리사이즈할 이미지 크기는 숫자를 바꿔보면서 테스트해야 함
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE)) #새로 배열 만들어서 이미지 리사이즈
plt.imshow(new_array, cmp = 'gray') 
plt.show() #새로 리사이즈된 이미지 보여줌 - 픽셀 형태로 출력됨

training_data = [] #이미지 트레이닝을 위한 배열

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #아까 설정해 둔 이미지 위치
        class_num = CATEGORIES.index(category) #분류할 카테고리에 인덱스, 즉 숫자로 이름을 붙여주기 위한 변수 초기화
        for img in os.listdir(path): #for문은 위에꺼 복붙
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array], class_num)
            except Exception as e:

create_training_data()
print(len(training_data))               

import random
random.shuffle(training_data) #학습하려는 이미지가 섞여있으니까 셔플

for sample in training[:10]:
    print(sample[1])

x = []
y = [] # 학습시키려는 데이터가 두 종류였기 때문에 x, y 두 개의 배열 초기화

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #이진화된 이미지라 마지막에 1을 적어줌

import pickle
#이미지 저장을 위해서
pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
 
pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in)
#여기까지 하면 이미지가 행렬로 저장됨
