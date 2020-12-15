# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:34:06 2018

@author: mir7942
"""

import os
import datetime
import time
import random
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
#from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

# 학습에 사용한 이미지는 (64, 64) 크기에 체널이 3개임
dims = (64, 64)
n_channels = 3
# 전체 데이터셋에서 사용할 시험 데이터셋의 비율
test_data_ratio = 0.2
# Epochs
num_epochs = 40
# Batch size
batch_size = 128
# 결과 저장 폴더
output_folder_path = "..\\Output"
# 데이터셋 폴더
dataset_folder_path = "D:/DataSet/VehicleDetection"

#%% GPU 개수를 알아낸다.
def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
#%% dataset_folder_path 폴더에서 데이터셋을 읽는다.
def load_dataset():
    # 자동차 이미지를 저장하기 위한 배열
    image_cars = []
    # 비-자동차 이미지를 저장하기 위한 배열
    image_notcars = []

    # DataSet 폴더에 있는 모든 png 파일명을 가져온다.
    image_filenames = glob.glob(dataset_folder_path + '/*/*/*.png')
    
    # 각각의 파일명에 대해...
    for image_filename in image_filenames:
        # 이미지 파일을 읽는다.
        image = mpimg.imread(image_filename)
        # 경로에 'non-vehicle'가 있으면 비-자동차로 인식한다.
        if 'non-vehicle' in image_filename:
            # 비-자동차 이미지를 image_notcars에 추가한다.
            image_notcars.append(image)
        else:
            # 자동차 이미지를 image_notcars에 추가한다.
            image_cars.append(image)
    
    # 이미지를 리턴한다.
    return image_cars, image_notcars

#%% 두 이미지를 화면에 출력한다.
def show_compare_images(image1, image2, image1_exp="Car", image2_exp="Not Car"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

#%% 데이터셋(data_x, data_y)을 학습 데이터셋(train_x, train_y)과 시험 데이터셋(test_x, test_y)으로 분리한다.
def partition_dataset(data_x, data_y):
    # 데이터셋의 크기
    m = data_x.shape[0]
    
    # 데이터셋을 섞는다.
    permutation = list(np.random.permutation(m))
    data_x = data_x[permutation, :]
    data_y = data_y[permutation]
    
    # 시험 데이터셋의 개수를 계산한다. 전체 데이터 중 test_data_ratio만큼 사용
    test_data_num = int(m * test_data_ratio)
        
    # 시험 데이터셋을 분리한다.
    test_x = data_x[:test_data_num]
    test_y = data_y[:test_data_num]
    # 학습 데이터셋을 분리한다.
    train_x = data_x[test_data_num : ]
    train_y = data_y[test_data_num : ]
        
    return train_x, train_y, test_x, test_y

#%% 모델 정의
def define_model():
    model = Sequential()    
        
    model.add(Conv2D(32, (5, 5), input_shape=(*dims, n_channels), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
        
    return model

#%% 학습 그래프 출력    
def plot_history(hist):
    plt.plot(hist.history['loss'], 'y', label='Train loss')
    plt.plot(hist.history['val_loss'], 'r', label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
        
    plt.plot(hist.history['acc'], 'b', label='Train acc')
    plt.plot(hist.history['val_acc'], 'g', label='Val acc')    
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')    
    plt.legend(loc='lower right')
    plt.show()
    
    return

#%% 메인 함수
def main():
    start_time = datetime.datetime.now()            
        
    # 실행할 때마다 랜덤값에 영향을 받지 않게 시드를 통일시킨다.
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(12345)
    tf.set_random_seed(1234)
        
    # GPU 사용 가능 확인
    num_gpu = len(get_available_gpus())
    use_gpu = num_gpu >= 2

    print("Train Car Classifier")
    print(start_time)
    
    # Dataset 폴더에서 자동차 이미지와 비-자동차 이미지를 읽어온다.
    image_cars, image_notcars = load_dataset()
    # 불러온 이미지 중 일부를 시험삼아 화면에 보여준다.
    show_compare_images(image_cars[0], image_notcars[0])
    
    # 이미지 데이터를 한 개의 배열로 합친다.
    data_x = np.vstack((image_cars, image_notcars)).astype(np.float64) 
    # 이미지 데이터에 해당하는 레이블을 만든다. 자동차는 1, 비-자동차는 0이다.
    data_y = np.hstack((np.ones(len(image_cars)), np.zeros(len(image_notcars))))
    
    # 이미지를 학습 데이터셋과 시험 데이터셋으로 분리한다.
    train_x, train_y, test_x, test_y = partition_dataset(data_x, data_y)
    
        
    m_train = train_x.shape[0] # 학습 데이터셋 개수
    m_test = test_x.shape[0] # 시험 데이터셋 개수   
        
    # 정보를 화면에 출력한다.
    print ("Number of GPUs: " + str(num_gpu))
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Number of epochs: " + str(num_epochs))
    print ("Batch size: " + str(batch_size))    
    print ("Image size: (" + str(dims[0]) + ", " + str(dims[1]) + ")")
    print ("Input channel: " + str(n_channels))
    
    # 모델 생성
    if use_gpu:
        with tf.device('/cpu:0'):
            model = define_model()
            parallel_model = multi_gpu_model(model, gpus=num_gpu)
    else:
        model = define_model()
    
    # 모델 컴파일
    if use_gpu:
        parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])        
    
    # output_folder_name이 없으면 폴더를 만든다.
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
            
    # 모델을 파일로 저장한다.
    with open(os.path.join(output_folder_path, 'model_architectures.json'), 'w') as f:
        f.write(model.to_json())
            
    # 콜벡 함수 지정
    '''
    callbacks = []    
        
    # 가장 좋은 가중치를 저장
    checkpoint_best_path = output_folder_path + '\\model_weights_best.h5'
    checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_best_only=True, save_weights_only=True, verbose=1)
    callbacks.append(checkpoint_best)
    '''
    # 모델 학습시키기
    print("Start Training")
    start_time = time.time()
    
    #history = parallel_model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs)
    
    # 데이터 강화를 위한 ImageDataGenerator 생성
    # 좌우 이동, 좌우 뒤집기, 확대/축소 적용
    # 20%를 검증용으로 사용
    datagen = ImageDataGenerator(width_shift_range=0.2,
                                 height_shift_range=0.2, 
                                 horizontal_flip=True,
                                 zoom_range=0.2,
                                 validation_split=0.2)
    
    # 데이터를 학습용과 검증용으로 분리
    train_generator = datagen.flow(train_x, train_y, batch_size = batch_size, subset='training')
    validation_generator = datagen.flow(train_x, train_y, batch_size = batch_size, subset='validation')

    # 모델 학습
    if use_gpu:
        history = parallel_model.fit_generator(train_generator,
                                               steps_per_epoch=train_x.shape[0] / batch_size * 10,
                                               epochs=num_epochs,
                                               validation_data=validation_generator,
                                               validation_steps=train_x.shape[0] / batch_size * 2)
    else:
        history = model.fit_generator(train_generator,
                                               steps_per_epoch=train_x.shape[0] / batch_size * 10,
                                               epochs=num_epochs,
                                               validation_data=validation_generator,
                                               validation_steps=train_x.shape[0] / batch_size * 2)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Training Finished!")
    print("Training Time(s): ", elapsed_time)    
    
    # 가중치를 저장한다.
    model.save_weights(os.path.join(output_folder_path, 'model_weights.h5'))
    
    # 학습 그래프 출력
    plot_history(history)
    
    # 시험셋을 이용해 모델 평가
    if use_gpu:
        score = parallel_model.evaluate(test_x, test_y)
    else:
        score = model.evaluate(test_x, test_y)
        
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
        
    return
    
if __name__ == "__main__":
    main()