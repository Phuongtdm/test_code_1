from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input, ZeroPadding3D, concatenate
from tensorflow.keras import regularizers, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from qpso import QPSO
import datetime
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier

# data_directory = './test_code_1/3D_data_total_np1.npz'

# with np.load(data_directory, allow_pickle=True) as data:
#     X_all = data['X_all']
#     Y_all = data['Y_all'] 

X_all = np.load('./test_code_1/X_all.npy', allow_pickle=True)
Y_all = np.load('./test_code_1/Y_all.npy', allow_pickle=True)

X_all = X_all / 255

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_test.shape)

# One-hot encode the target labels
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Convert the target labels to dense tensors
y_train_dense = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)
y_test_dense = tf.convert_to_tensor(y_test_onehot, dtype=tf.float32)

def model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, pol1, pol2, pol3,dropout, dense):
    nb_filter1  = filt1;
    nb_filter2  = filt2;
    nb_filter3  = filt3;
    filter1_size = filtlen_1;
    filter2_size = filtlen_2;
    filter3_size = filtlen_3;
    poling1 = pol1;
    poling2 = pol2;
    poling3 = pol3;
    dropout = dropout;
    dense = dense;

    conv1 = Conv3D(filt1, (filtlen_1, filtlen_1, filtlen_1), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(pol1, pol1, pol1), padding="same")(conv1)
    ave1 = AveragePooling3D(pool_size=(pol1, pol1, pol1), padding="same")(conv1)
    comb1 = concatenate([max1, ave1])
    
    conv2 = Conv3D(filt2, (filtlen_2, filtlen_2, filtlen_2), activation='relu', padding="same")(comb1)
    max2 = MaxPooling3D(pool_size=(pol2, pol2, pol2), padding="same")(conv2)
    ave2 = AveragePooling3D(pool_size=(pol2, pol2, pol2), padding="same")(conv2)
    comb2 = concatenate([max2, ave2])
    
    conv3 = Conv3D(filt3, (filtlen_3, filtlen_3, filtlen_3), activation='relu', padding="same")(comb2)
    max3 = MaxPooling3D(pool_size=(pol3, pol3, pol3), padding="same")(conv3)
    ave3 = AveragePooling3D(pool_size=(pol3, pol3, pol3), padding="same")(conv3)
    comb3 = concatenate([max3, ave3])
    
    flat = Flatten()(comb3)
    drop = Dropout(dropout)(flat)
    dense_out = Dense(dense, activation='relu')(drop)
    
    return dense_out

def cnn(x):
    t1_0 = datetime.datetime.now()
    filt1 = int(x[0])
    filt2 = int(x[1])
    filt3 = int(x[2])
    filtlen_1 = int(x[3])
    filtlen_2 = int(x[4])
    filtlen_3 = int(x[5])
    pol1 = int(x[6])
    pol2 = int(x[7])
    pol3 = int(x[8])
    dropout = x[9]
    dense = int(x[10])

    inp_1 = Input(shape=(256, 256, 5, 1))
    zeropad_1 = ZeroPadding3D(padding=(1, 1, 1))(inp_1)
    
    nn = model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, pol1, pol2, pol3, dropout, dense)
    
    output = Dense(10, activation='softmax')(nn)

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    cnn_history = cnn_model.fit(X_train, y_train_dense, epochs=100, batch_size=16, verbose=0, validation_data=(X_test, y_test_dense))
    
    score = cnn_history.history['accuracy'][-1]

    t2_0 = datetime.datetime.now()
    elapsed_time = t2_0 - t1_0

    log(s, score, elapsed_time)

    return -score

def log(s, score, elapsed_time):
    best_value = [p.best_value for p in s.particles()]
    best_value_avg = np.mean(best_value)
    best_value_std = np.std(best_value)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open('optimization_log.txt', 'a') as f:
        f.write("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}  {4}  {5}\n".format(
            str(s.gbest), score, best_value_avg, best_value_std, current_time, elapsed_time))

bounds = [(8, 32), (8, 32), (8, 32), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (0.2, 0.5), (32, 128)]
NDim = 11
NParticle = NDim * 3
MaxIters = 50

s = QPSO(cnn, NParticle, NDim, bounds, MaxIters)
s.update(callback=log, interval=100)

with open('optimization_log.txt', 'a') as f:
    f.write("Found best position: {0}\n".format(s.gbest))
