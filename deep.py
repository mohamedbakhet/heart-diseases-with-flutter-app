import numpy as np

from keras.models import model_from_json
from keras.optimizers import Adam
import keras
from sklearn.preprocessing import LabelEncoder
import os
import wfdb
from wfdb import processing
import scipy.signal as sign
from ecgdetectors import Detectors
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Conv1D
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
#from tensorflow.keras import datasets, layers, models
class model:
    # def __init__(self):
    #     json_file = open('ResNetNaive.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     self.loaded_model = model_from_json(loaded_model_json)
    #     self.loaded_model.load_weights("ResNetNaive.h5")
    #     opt = Adam(lr=0.001)
    #     self.loaded_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #
    #     json_file1 = open('incML.json', 'r')
    #     loaded_model_json1 = json_file1.read()
    #     json_file1.close()
    #     self.loaded_model1 = model_from_json(loaded_model_json1)
    #     self.loaded_model1.load_weights("incMLs.h5")
    #     opt = Adam(lr=0.001)
    #     self.loaded_model1.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])




    def de_noise(self,fs,sig):
        low_cut = (1) / (fs)
        high_cut = (40) / (fs)
        a, b = sign.butter(2, [low_cut, high_cut], 'pass')
        f = sign.filtfilt(a, b, sig, axis=0)
        return f

    def read(self,path):
        sig, fi = wfdb.rdsamp(path)

        return sig, fi

    def dynamic_segmentation(self,fs, sig):
        detectors = Detectors(fs)
        l_peaks = []
        l_pqrst = []
        for i in sig:
            r_peaks = detectors.pan_tompkins_detector(i)
            end = len(i) - 1
            l = []
            for i in range(len(r_peaks)):
                if i == 0:
                    continue
                if i == len(r_peaks) - 1:
                    continue
                rp = r_peaks[i] - r_peaks[i - 1]
                rn = r_peaks[i + 1] - r_peaks[i]
                before = (1 / 3) * max(rp, rn)
                after = (2 / 3) * max(rp, rn)
                rbefore = r_peaks[i] - before
                rafter = r_peaks[i] + after
                l.append(int(rbefore))
                l.append(int(rafter))
            l_peaks.append(r_peaks)
            l_pqrst.append(l)
        return l_peaks, l_pqrst

    def to_one_dict(self,dict):
        l = {}
        for key, value in dict.items():
            if key == 'comments':
                for i in value:
                    ls = i.split(":")
                    l[ls[0]] = ls[1]

            else:
                l[key] = value
        return l

    def build_header_file(self,p_fi):
        i = self.to_one_dict(p_fi)
        del i['sig_name']
        del i['units']
        df = pd.DataFrame([i])
        return df

    def cuts(self,f, pqrst, sign_name):
        list_peats = {}
        for l in range(len(f)):
            peats = []
            i = 0
            while i < len(pqrst[l]) - 1:
                si = f[l][pqrst[l][i]:pqrst[l][i + 1]]
                samp = sign.resample(si, 300)
                peats.append(samp)
                i += 2
            list_peats[sign_name[l]] = peats
        return list_peats

    def normalize(self,f):
        ft = f.T
        for i in range(len(ft)):
            ft[i] = (ft[i] - min(ft[1])) / (max(ft[1]) - min(ft[1]))
        return ft.T

    def get_name(self,folder, lis):
        folder = folder + "/"
        l = []
        l_out = []
        for i in lis:
            st = i[:-4]
            if (st not in l):
                l.append(st)
        for i in l:
            l_out.append(folder + i)
        return l_out

    def processing(self,path):
        sig, fi = wfdb.rdsamp(path)
        fs = fi['fs']
        f = self.de_noise(fs, sig)
        f = self.normalize(f)
        ft = f.T
        # df = pd.DataFrame(f, columns=fi['sig_name'])
        y, z = self.dynamic_segmentation(fs, ft)
        head = self.build_header_file(fi)
        peats = self.cuts(ft, z, fi['sig_name'])
        return peats, head, fi

    def col(self,peats):
        f = 0
        min = len(peats['i'])
        for key, value in peats.items():
            t = len(value)
            if t < min:
                min = t

        leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
        for i in leads:

            arr = np.array(peats[i])
            arr=arr[:min]
            if f == 0:
                lis = arr.reshape(-1, 300, 1)
                f = 1
            else:
                lis2 = arr.reshape(-1, 300, 1)
                lis = np.concatenate((lis, lis2), axis=2)
        return lis
    loaded_model=None
    loaded_model1=None




    def predict(self,x):
        # model = Sequential()
        # '''model = models.Sequential()
        # model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(300, 15,1)))
        # model.add(layers.MaxPooling2D((5,5 )))
        # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        # model.add(layers.MaxPooling2D((5, 5)))
        # model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        # model.add(layers.MaxPooling2D((5,5 )))
        # model.add(layers.Conv2D(256, (5, 5), activation='relu'))
        # model.add(layers.MaxPooling2D((5, 5)))
        # model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        # model.add(layers.MaxPooling2D((5, 5)))'''
        # tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
        conv_input = input_data(shape=[None, 300, 15, 1], name='input')
        conv0 = conv_2d(conv_input, 32, 5, activation='relu')
        pool0 = max_pool_2d(conv0, 5)

        conv1 = conv_2d(pool0, 64, 5, activation='relu')
        pool1 = max_pool_2d(conv1, 5)

        conv2 = conv_2d(pool1, 128, 5, activation='relu')
        pool2 = max_pool_2d(conv2, 5)

        conv3 = conv_2d(pool2, 256, 5, activation='relu')
        pool3 = max_pool_2d(conv3, 5)

        conv4 = conv_2d(pool3, 128, 5, activation='relu')
        pool4 = max_pool_2d(conv4, 5)

        conv5 = conv_2d(pool4, 64, 5, activation='relu')
        pool5 = max_pool_2d(conv5, 5)

        fully_layer = fully_connected(pool5, 1024, activation='relu')
        fully_layer = dropout(fully_layer, 0.5)

        cnn_layers = fully_connected(fully_layer, 15, activation='softmax')

        cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                                name='targets')
        loaded_model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

        loaded_model.load('cnn.h5')
        # peats, df, fi = pre.processing(x)
        # X_test = pre.col(peats)
        # print (X_test.shape)
        # X_test = np.array([i for i in X_test]).reshape(-1, 300, 15,1)
        # print (X_test.shape)
        # fi = pre.to_one_dict(fi)
        # print(fi)
        # diasease = fi['Reason for admission']
        # d = ['Healthy control', 'Bundle branch block', 'Cardiomyopathy', 'Dysrhythmia', 'Myocarditis', 'Palpitation',
        #      'Stable angina', 'Unstable angina', 'Heart failure (NYHA 2)', 'Heart failure (NYHA 3)',
        #      'Valvular heart disease',
        #      'Myocardial infraction', 'Heart failure (NYHA 4)', 'Hypertrophy']

        d = [['Healthy control', 0], ['Bundle branch block', 1], ['Cardiomyopathy', 2], ['Dysrhythmia', 3],
             ['Myocarditis', 4], ['Palpitation', 5], ['Stable angina', 6], ['Unstable angina', 7],
             ['Heart failure (NYHA 2)', 8], ['Heart failure (NYHA 3)', 9], ['Valvular heart disease', 10],
             ['Myocardial infraction', 11], ['Heart failure (NYHA 4)', 12], ['Hypertrophy', 13], ['normal', 14]]
        df_names = pd.DataFrame(d, columns=['class', 'Label'])

        # json_file = open('ResNetNaive.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # loaded_model.load_weights("ResNetNaive.h5")
        # opt = Adam(lr=0.001)
        # loaded_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # y_predict = model.predict(X_test)
        # y_predict = y_predict[0].argmax()
        # input = df_names.loc[df_names['Label'] == y_predict]['class'].values[0]

        # labels = LabelEncoder()
        # train_labels = labels.fit(d)
        # first = labels.inverse_transform(y_predict)[0]
        #print(self.first)
        # return input
        peats, df, fi = self.processing(x)
        X_test = self.col(peats)
        print(X_test.shape)
        X_test = X_test.reshape(-1, 300, 15, 1)
        print(X_test.shape)
        fi = self.to_one_dict(fi)
        print(fi)
        # diasease = fi['Reason for admission']
        # d = ['Healthy control', 'Bundle branch block', 'Cardiomyopathy', 'Dysrhythmia', 'Myocarditis', 'Palpitation',
        #      'Stable angina', 'Unstable angina', 'Heart failure (NYHA 2)', 'Heart failure (NYHA 3)',
        #      'Valvular heart disease',
        #      'Myocardial infraction', 'Heart failure (NYHA 4)', 'Hypertrophy','normal']
        # json_file = open('ResNetNaive.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # loaded_model.load_weights("ResNetNaive.h5")
        # opt = Adam(lr=0.001)
        # loaded_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # y_predict = loaded_model.predict(X_test)
        # y_predict = y_predict.argmax(axis=1)
        #loaded_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        y_predict = loaded_model.predict(X_test)
        y_predict = y_predict[0].argmax()
        input = df_names.loc[df_names['Label'] == y_predict]['class'].values[0]
        # labels = LabelEncoder()
        # train_labels = labels.fit(d)
        # first =labels.inverse_transform(y_predict)[0]
        # print(self.first)
        return input

    # def predictmyco(self,x):
    #     k = ['anterior', 'antero-lateral', 'antero-septal', 'antero-septo-lateral', 'inferior', 'infero-lateral',
    #          'infero-postero-lateral', 'lateral', 'posterior', 'postero-lateral', 'infero-posterior']
    #     peats, df, fi = self.processing(x)
    #     X_test = self.col(peats)
    #     X_test = X_test.reshape(-1, 300, 15)
    #     fi = self.to_one_dict(fi)
    #     print(fi)
    #     diasease = fi['Acute infarction (localization)']
    #     # json_file = open('vggML.json', 'r')
    #     # loaded_model_json = json_file.read()
    #     # json_file.close()
    #     # loaded_model = model_from_json(loaded_model_json)
    #
    #     img_input = Input((300, 15))
    #     # Block 1
    #     x = layers.Conv1D(64, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block1_conv1')(img_input)
    #     x = layers.Conv1D(64, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block1_conv2')(x)
    #     x = layers.MaxPooling1D(2, strides=2, name='block1_pool', padding='same')(x)
    #
    #     # Block 2
    #     x = layers.Conv1D(128, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block2_conv1')(x)
    #     x = layers.Conv1D(128, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block2_conv2')(x)
    #     x = layers.MaxPooling1D(2, strides=2, name='block2_pool', padding='same')(x)
    #
    #     # Block 3
    #     x = layers.Conv1D(256, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block3_conv1')(x)
    #     x = layers.Conv1D(256, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block3_conv2')(x)
    #     x = layers.Conv1D(256, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block3_conv3')(x)
    #     x = layers.MaxPooling1D(2, strides=2, name='block3_pool', padding='same')(x)
    #
    #     # Block 4
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block4_conv1')(x)
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block4_conv2')(x)
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block4_conv3')(x)
    #     x = layers.MaxPooling1D(2, strides=2, name='block4_pool', padding='same')(x)
    #
    #     # Block 5
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block5_conv1')(x)
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block5_conv2')(x)
    #     x = layers.Conv1D(512, 3,
    #                       activation='relu',
    #                       padding='same',
    #                       name='block5_conv3')(x)
    #     x = layers.MaxPooling1D(2, strides=2, name='block5_pool', padding='same')(x)
    #
    #     # Classification block
    #     x = layers.Flatten(name='flatten')(x)
    #     x = layers.Dense(128, activation='relu', name='fc1')(x)  # reduced dim for 1-d task
    #     x = layers.Dense(128, activation='relu', name='fc2')(x)
    #     x = layers.Dense(14, activation='softmax', name='predictions')(x)
    #
    #     # Create model.
    #     loaded_model = models.Model(img_input, x, name='vgg16')
    #
    #
    #     loaded_model.load_weights("vggML.h5")
    #     opt = Adam(lr=0.001)
    #     loaded_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #     y_predict = loaded_model.predict(X_test)
    #     print(y_predict)
    #     y_predict =y_predict.argmax(axis=1)
    #     labels = LabelEncoder()
    #     train_labels = labels.fit(k)
    #     second = labels.inverse_transform(y_predict)[0]
    #     return second
