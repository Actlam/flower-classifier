from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys
import numpy as np
from PIL import Image
from sklearn import model_selection


classes = ["Dandelion", "DandelionFluffy", "Hydrangea", "RedSpiderLily"]
num_classes = len(classes)
image_size = 50

def build_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(50,50,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation('relu'))                   # 活性化関数
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))           
    model.add(MaxPooling2D(pool_size=(2, 2)))       # 一番大きい値を取り出し, 特徴を際立たせる
    model.add(Dropout(0.25))                         # 25%カットして計算の偏りを減らす
    
    model.add(Flatten())                            # データを一列に並べる
    model.add(Dense(512))                                # 全結合層
    model.add(Activation('relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(4)) 
    model.add(Activation('softmax'))               # 画像との一致率を足すて1になるよう調整
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    # モデルのロード
    model = load_model('./flower_cnn.h5')           # kerasの関数
    
    return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()
    
    result = model.predict([X])[0]
    predicted = result.argmax()                         # 一番大きい配列の添え字を返す
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()
    
    