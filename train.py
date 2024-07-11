from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import Callback

class ProgressCallback(Callback):
    def __init__(self, progress_signal):
        super().__init__()
        self.progress_signal = progress_signal

    def on_epoch_end(self, epoch, logs=None):
        self.progress_signal.emit(epoch + 1)

def train_model(img_width, img_height, img_dir, epochs, model_dir, model_name, progress_signal):
    test_person = ""
    image_w = img_width
    image_h = img_height

    train = 'train'
    categories = []
    accident_dir = img_dir

    if not os.path.exists(accident_dir):
        print(f"Error: Directory {accident_dir} does not exist.")
        return

    my_dirs = [d for d in os.listdir(accident_dir) if os.path.isdir(os.path.join(accident_dir, d))]
    for person in my_dirs:
        test_person = person
        categories.append(person)

    nb_classes = len(categories)
    pixels = image_w * image_h * 3
    X = []
    Y = []

    for idx, cat in enumerate(categories):
        label = [0 for i in range(nb_classes)]
        label[idx] = 1
        image_dir = os.path.join(accident_dir, cat)
        files = glob.glob(image_dir + "/*.png") + glob.glob(image_dir + "/*.jpg")
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    X_train = X_train.astype("float32") / 256
    X_test = X_test.astype("float32") / 256

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model_path = model_dir
    model_name = "/" + model_name + ".hdf5"
    hdf5_file = model_path + model_name

    if train == "load":
        if os.path.exists(hdf5_file):
            model.load_weights(hdf5_file)
        else:
            print(f"Error: File {hdf5_file} does not exist.")
            return
    else:
        progress_callback = ProgressCallback(progress_signal)
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, callbacks=[progress_callback])
        model.save(hdf5_file)

    score = model.evaluate(X_test, y_test)
    print('loss=', score[0])  # loss
    print('accuracy=', score[1])  # accuracy

    test_image = './img01/normal_01/chip_r_01.png'

    if not os.path.exists(test_image):
        print(f"Error: Test image {test_image} does not exist.")
        return

    print('Test person is ' + test_person + ', test image is ' + test_image)

    img = Image.open(test_image)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float32") / 256
    X = X.reshape(-1, image_w, image_h, 3)

    pred = model.predict(X)
    result = [np.argmax(value) for value in pred]  # 예측 결과에서 클래스 추출
    print('New data category : ', categories[result[0]])

    print(pred)
