import keras
from keras import layers
from keras.optimizers import SGD
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers

dataset='drugbank'
data='t'   #  i/t
Nets = []
input_dims = []
path_to_string_nets = './data2/'+dataset+'/'+dataset+"_"+data
string_nets=[
    '1_ingredient_AP_TC_similarity_AllSimScores',
    '2_ingredient_EC4_TC_similarity_AllSimScores',
    '3_ingredient_EC6_TC_similarity_AllSimScores',
    '4_ingredient_FC4_TC_similarity_AllSimScores',
    '5_ingredient_FC6_TC_similarity_AllSimScores',
    '6_ingredient_MACCS_TC_similarity_AllSimScores',
    '7_ingredient_RDK_TC_similarity_AllSimScores',
    '8_ingredient_TOPTOR_TC_similarity_AllSimScores',
 ]
filenames= []
for net in string_nets:
    filenames.append(path_to_string_nets +net + '.mat')
print ("### Loading drug network ---" ,filenames[0])
N= sio.loadmat(filenames[0], squeeze_me=True)
Net = N['aveNets']
print("num of fei 0 = ",np.count_nonzero(Net))
Nets.append(minmax_scale(Net))
input_dims.append(Net.shape[0])


train, test = train_test_split(Net, test_size=0.2)
train_data = train
test_data = test
#denoised block
noise_factor = 0.6
std=1.0
X_train_noisy = train.copy()
X_test_noisy = test.copy()
X_train_noisy = X_train_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=train.shape)
X_test_noisy = X_test_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=test.shape)
X_train_noisy = np.clip(X_train_noisy, 0, 1)
X_test_noisy = np.clip(X_test_noisy, 0, 1)
x_train = X_train_noisy.astype('float32')
x_test = X_test_noisy.astype('float32')
print('dim=',x_train.shape[1])

###  AE-1（stacked block）
input_1 = keras.Input(shape=(x_train.shape[1],))

##AE
encoded_1 = layers.Dense(400, activation='relu',
               activity_regularizer=regularizers.l1(10e-6))(input_1)  #sparse block
decoded_1 = layers.Dense(x_train.shape[1], activation='sigmoid')(encoded_1)
autoencoder_1 = keras.Model(input_1, decoded_1)

print('-------')
print (autoencoder_1.summary())

encoder_1 = keras.Model(input_1, encoded_1)
sgd = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
autoencoder_1.compile(optimizer=sgd, loss='mse')

history = autoencoder_1.fit(x_train, train,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

###  AE-2
hid_1=encoder_1.predict(Nets[0])
train_hid_1= encoder_1.predict(x_train)
test_hid_1= encoder_1.predict(x_test)


input_2 = keras.Input(shape=(400,))
encoded_2 = layers.Dense(200, activation='relu',
               activity_regularizer=regularizers.l1(10e-6))(input_2)
decoded_2 = layers.Dense(400, activation='sigmoid')(encoded_2)

autoencoder_2 = keras.Model(input_2, decoded_2)
print('-------')
print (autoencoder_2.summary())

encoder_2 = keras.Model(input_2, encoded_2)
sgd = SGD(lr=0.00015, momentum=0.9, decay=0.0, nesterov=False)
autoencoder_2.compile(optimizer=sgd, loss='mse')
history = autoencoder_2.fit(train_hid_1, train_hid_1,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(test_hid_1, test_hid_1))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#AE-3
hid_2=encoder_2.predict(hid_1)
train_hid_2= encoder_2.predict(train_hid_1)
test_hid_2= encoder_2.predict(test_hid_1)


input_3 = keras.Input(shape=(200,))
encoded_3 = layers.Dense(25, activation='relu',
               activity_regularizer=regularizers.l1(10e-6))(input_3)
decoded_3 = layers.Dense(200, activation='sigmoid')(encoded_3)

autoencoder_3 = keras.Model(input_3, decoded_3)
print('-------')
print (autoencoder_3.summary())

encoder_3 = keras.Model(input_3, encoded_3)
sgd = SGD(lr=0.00015, momentum=0.9, decay=0.0, nesterov=False)
autoencoder_3.compile(optimizer=sgd, loss='mse')
history = autoencoder_3.fit(train_hid_2, train_hid_2,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(test_hid_2, test_hid_2))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

features= encoder_3.predict(hid_2)
print('feature-dim=',features.shape)
features = minmax_scale(features)
sio.savemat('./data3/' + dataset + '/' + dataset +'_CWI_'+data+'.mat',{'features': features})

