# -*- coding: utf-8 -*-

import os
#import glob
#import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2

# 모델 저장 파일 위치
output_folder_name = "..\\Output"
# 테스트 이미지
test_image_path = "..\\TestImages\\test1.jpg"
# 학습 이미지 크기
dims = (64, 64)
# 이미지 체널 수
n_channels = 3
# 윈도우 이동 간격
step_size = 16
# 윈도우 크기    
window_size = dims
# 최소 이미지 크기
min_size = window_size
# 이미지 축소 비율
downscale = 1.2
# NMS threshold
threshold = 0.4

'''
#%% 디버깅:폴더에서 임의의 이미지를 읽는다.
def load_test_images():
    # 자동차 이미지를 저장하기 위한 배열
    image_filename_cars = []
    # 비-자동차 이미지를 저장하기 위한 배열
    image_filename_notcars = []

    # DataSet 폴더에 있는 모든 png 파일명을 가져온다.
    image_filenames = glob.glob('../DataSet/*/*/*.png')
    
    # 각각의 파일명에 대해...
    for image_filename in image_filenames:
        # 경로에 'non-vehicle'가 있으면 비-자동차로 인식한다.
        if 'non-vehicle' in image_filename:
            # 비-자동차 파일명을 image_filename_notcars에 추가한다.
            image_filename_notcars.append(image_filename)
        else:
            # 자동차 파일명을 image_filename_notcars에 추가한다.
            image_filename_cars.append(image_filename)
            
    # 파일을 임의로 한 개 골라 이미지를 읽는다.
    image_car = mpimg.imread(random.choice(image_filename_cars))
    image_notcar = mpimg.imread(random.choice(image_filename_notcars))
        
    # 이미지를 리턴한다.
    return image_car, image_notcar

#%% 디버깅: 모델을 이용해 예측한 결과를 보여준다.
def show_prediction(model, image):
    # 화면에 이미지를 보여준다.
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    # 결과를 예측한다.
    predicted = model.predict(image.reshape(1, *dims, n_channels))
    # 예측한 결과를 출력한다.
    if predicted >= 0.5:
        print("Predicted = Car")
    else:
        print("Predicted = Not a Car")    
    
    return
'''

#%% 축소한 이미지를 yield를 이용해 반복자로 반환한다.
def pyramid(image, downscale=1.5, min_size=(64, 64)): 
    # 원래 이미지를 yield한다.
    yield image 
 
    # 단계적으로 이미지를 축소하고, yield한다.
    while True: 
        # 축소할 이미지 크기를 계산한다.
        w = int(image.shape[1] / downscale) 
        # 이미지 크기를 줄인다.
        image = resize(image, width=w) 
 
        # 축소한 이미지 크기가 min_size보다 작으면 멈춘다.
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]: 
            break 
 
        # 축소한 이미를 yield한다.
        yield image 

#%% 이미지 크기를 줄인다.
def resize(image, width=None, height=None, inter=cv2.INTER_AREA): 
    # 이미지 크기
    (h, w) = image.shape[:2] 

    # width와 height가 모두 None이면 원본 이미지를 반환한다.
    if width is None and height is None: 
        return image 

    # width가 None이면...
    if width is None: 
        # height에 맞게 width를 계산한다.
        r = height / float(h) 
        dim = (int(w * r), height) 
    # height가 None이면...
    else: 
        # width에 맞게 height를 계산한다.
        r = width / float(w) 
        dim = (width, int(h * r)) 

    # OpenCV를 이용해 이미지 크기를 조정한다.
    resized = cv2.resize(image, dim, interpolation=inter) 
     
    return resized

#%% 슬라이딩 윈도우를 yield를 이용해 반복자로 반환한다.
def sliding_window(image, step_size, window_size):
	# image 전체에 대해 가로, 세로 방향으로 일정한 step_size 만큼 이동한다.
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# 현재 윈도우 위치(x,y)에서 window_size만큼의 부분 이미지를 반환한다.
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


#%% NMS 구현
def non_max_suppression(boxes, overlap_thresh=0.7):
    # 박스가 없으면 종료
    if len(boxes) == 0:
        return []

    # boxes가 정수면 float로 변환. 나중에 나눗셈 연산을 정확하게 하기 위해
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []

    # boxes들의 각 좌표값
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 각 box 면적을 계산
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # box를 정렬한다.
    idxs = np.argsort(y2)

    # 각각의 box에 대해서...
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 겹치는 부분을 계산한다.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 겹치는 부분의 크기를 계산
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 겹치는 부분의 면적 비율을 계산
        overlap = (w * h) / area[idxs[:last]]

        # 겹치는 부분이 overlap_thresh 이상이면 지운다. 즉, 다른 박스와 겹치기 때문에 없어도 된다.
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # 남아있는 박스를 반환한다.
    return boxes[pick].astype("int")            
            
#%% 메인 함수
def main():
    # 모델을 불러온다.
    with open(os.path.join(output_folder_name, 'model_architectures.json'), 'r') as f:
        model = model_from_json(f.read())
    
    # 가중치를 불러온다.
    model.load_weights(os.path.join(output_folder_name, 'model_weights.h5'))    
    # 모델을 컴파일한다.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    
    # 디버깅용: 학습 모델을 테스트 한다.
    # 모델을 표시한다.
    #model.summary()    
    # 모델 테스트를 위해서 Dataset 폴더에서 자동차 이미지와 비-자동차 이미지를 임의로 한 개 읽어온다.
    #image_car, image_notcar = load_test_images()
    #image_notcar = mpimg.imread('..\\test.png')
    #image_notcar.reshape(1, *dims, n_channels)
    # 불러온 image_car 이미지를 이용해 예측한다.
    #print("Actual = Car")
    #show_prediction(model, image_car)
    #print("Actual = Not a Car")
    #show_prediction(model, image_notcar)    
    
    # 테스트용 이미지를 불러온다.
    test_image = mpimg.imread(test_image_path)
    # 분류기가 0~1 스케일을 사용하였기 때문에 이미지 스케일을 0~255에서 0~1로 조정한다.
    test_image = test_image.astype(np.float32) / 255
    
    # 이미지 처리 후 저장할 파일 이름
    path, filename = os.path.split(test_image_path)
    filename = os.path.splitext(filename)[0]
    test_image_before_nms_path = os.path.join(path, filename + '_before_nms.png') 
    test_image_after_nms_path = os.path.join(path, filename + '_after_nms.png') 
    
    # 화면에 이미지를 보여준다.    
    plt.imshow(test_image)
    plt.title('Original image')
    plt.xticks([]), plt.yticks([])
    plt.show()    
        
    # 검색 결과를 저장할 리스트. 박스의 양 끝 좌표(x1,y1,x2,y2)가 저장됨
    detections = []
    # downscale 조정 인자
    downscale_power = 0
    # 이미지 복사본
    test_image_clone = test_image.copy()
    # 이미지 피라미드를 이용해 이미지를 단계적으로 축소시킨다.
    for scaled_image in pyramid(test_image, downscale, min_size):
        # 슬라이딩 윈도우 내의 부분 이미지에 적용한다.
        for (x, y, window) in sliding_window(scaled_image, step_size, window_size):
            #크기가 맞지 않는다면 무시한다. 이미지는 (세로 크기, 가로 크기)로 저장됨
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            # 분류기를 이용해 자동치인지 예측한다.
            predicted = model.predict(window.reshape(1, *dims, n_channels))
            # 자동차라고 예측했다면...
            if predicted >= 0.5:
                # 축소 전의 이미지 상 위치를 계산한다.
                x1 = int(x * (downscale ** downscale_power)) 
                y1 = int(y * (downscale ** downscale_power)) 
                # 윈도우 네 모서리 좌표를 detections에 저장한다.
                detections.append((x1, y1, 
                                   x1 + int(window_size[0] * (downscale ** downscale_power)),
                                   y1 + int(window_size[1] * (downscale ** downscale_power))))
        
        # downscale 조정 인자를 조절한다.
        downscale_power += 1 
        
    # 찾은 윈도우들을 보여준다.
    test_image_before_nms = test_image_clone.copy() 
    for (x1, y1, x2, y2) in detections:
        # 이미지에 윈도우를 그린다.
        cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (0, 1, 0), thickness=2) 
    
    # 화면에 이미지를 표시한다.
    plt.title('Detected cars befor NMS')    
    plt.imshow(test_image_before_nms)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imsave(test_image_before_nms_path, test_image_before_nms)
    
    # Non-Maxima Suppression을 수행한다.
    detections = non_max_suppression(np.array(detections), threshold) 
    test_image_after_nms = test_image_clone 
    for (x1, y1, x2, y2) in detections: 
        # 이미지에 윈도우를 그린다.
        cv2.rectangle(test_image_after_nms, (x1, y1), (x2, y2), (0, 1, 0), thickness=2) 

    # 화면에 이미지를 표시한다.
    plt.title('Detected cars after NMS')
    plt.imshow(test_image_after_nms)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imsave(test_image_after_nms_path, test_image_after_nms)    

    return
    
if __name__ == "__main__":
    main()