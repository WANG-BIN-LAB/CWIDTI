import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import SGD
from sklearn.metrics import precision_recall_curve,auc,roc_curve
import warnings
import sklearn.exceptions
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from imblearn.over_sampling import SMOTE
import scipy.io
from keras.layers import BatchNormalization
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
from keras import regularizers
def loadDatadet(infile):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        # pdb.set_trace()
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    return dataset
def cross_validation(intMat, seeds, cv=0,n_fold=5):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size//n_fold
        for i in range(n_fold):
            if i < n_fold-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data
def build_model():
    model = Sequential()
    model.add(Dense(input_dim=(len(DT_feature[0])), output_dim=300, init='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=300, output_dim=200, init='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=200, output_dim=100, init='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=100, output_dim=1, init='he_normal',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('sigmoid'))
    # adadelta=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=sgd ,metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    dataset = "drugbank"
    Batchsize =1024
    Epoch =100
    learn_rate = 0.001
    oversample = 1
    n_fold=5
    predict_num=100
    InPath = "./data/" + dataset +'/interaction.txt'
    DFPath = "./data3/" + dataset +'/'+dataset+ "_CWI_d.mat"
    TFPath = "./data3/" + dataset +'/'+dataset+ "_CWI_t.mat"
    data = np.loadtxt(InPath)
    drug_num, target_num = data.shape
    d_feature = scipy.io.loadmat(DFPath)
    t_feature = scipy.io.loadmat(TFPath)
    D_feature = np.array(d_feature['features'])
    T_feature = np.array(t_feature['features'])
    DT_feature = np.zeros(shape=(len(D_feature) * len(T_feature), len(D_feature[0]) + len(T_feature[0])))
    DT_Feature = np.zeros(shape=(len(D_feature) * len(T_feature), len(D_feature[0]) + len(T_feature[0])))
    h = 0
    k = 0
    for i in range(0, len(D_feature)):
        for j in range(0, len(T_feature)):
            DT_feature[h] = np.concatenate((D_feature[i], T_feature[j]))
            DT_Feature[k] = DT_feature[0]
            k = k + 1
    x_train = DT_Feature
    seeds = [1,2,3,4,5]
    se = len(seeds)
    mean_AUCs = [0 for _ in range(se)]
    mean_AUPRs = [0 for _ in range(se)]
    all_AUPR = [[0] * n_fold for _ in range(se)]
    all_AUC = [[0] * n_fold for _ in range(se)]

    mode = "T"
    print('S3:')
    pair = False
    data = np.transpose(data)
    cv_data = cross_validation(data, seeds, 0, n_fold)
    resultpath = "./result/"+dataset+'/'+dataset+"_CWM-DTI_S3.txt"

    runtimes = 0
    for seed in seeds:
        print('run #', runtimes + 1)
        test_index = [0 for _ in range(n_fold)]
        fold = 0
        for i in range(0, n_fold):
            print('fold #', fold + 1)
            test_index[i] = cv_data[seed][i][1]
            pdb.set_trace()
            test_feature = []
            test_label = []
            test_hang = []
            train_feature = []
            train_label = []
            for a in range(0, len(test_index[i])):
                m = test_index[i][a][0]
                la = test_index[i][a][1]
                test_hang.append(m)
                test_feature.append(x_train[target_num * la + m])
                test_label.append(cv_data[seed][i][2][a])
            hangset = set(test_hang)
            test_hang = list(hangset)
            train_hang = []
            for h_train in range(0, target_num):
                if h_train not in test_hang:
                    train_hang.append(h_train)
                    for ln in range(0, drug_num):
                        train_label.append(data[h_train][ln])
                        train_feature.append(x_train[target_num * ln + h_train])
            test_label = np.array(test_label)
            test_feature = np.array(test_feature)
            train_label = np.array(train_label)
            train_feature = np.array(train_feature)
            model_smote = SMOTE()
            test_feature_resampled, test_label_resampled = model_smote.fit_sample(test_feature, test_label)
            train_feature_resampled, train_label_resampled = model_smote.fit_sample(train_feature, train_label)
            earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5)
            my_model = build_model()
            history = my_model.fit(train_feature_resampled, train_label_resampled, epochs=Epoch,batch_size=Batchsize, verbose=True,callbacks=[earlystop], validation_data=(test_feature_resampled, test_label_resampled))
            val_pre_pro = my_model.predict_proba(test_feature_resampled, batch_size=Batchsize, verbose=False)
            val_pre_class = my_model.predict_classes(test_feature_resampled, batch_size=Batchsize, verbose=False)
            fpr, tpr, auc_thresholds = roc_curve(test_label_resampled, val_pre_pro, pos_label=1)
            auc_score = auc(fpr, tpr)
            precision, recall, pr_threshods = precision_recall_curve(test_label_resampled, val_pre_pro, pos_label=1)
            aupr_score =auc(recall, precision)
            all_AUPR[runtimes][i] = aupr_score
            all_AUC[runtimes][i] = auc_score
            fold = fold+1
            print('Data=', dataset)
            print('AUC=', auc_score)
            print('AUPR=' ,aupr_score)
        mean_AUCs[runtimes] = np.mean(np.array(all_AUC[runtimes]))
        mean_AUPRs[runtimes] = np.mean(np.array(all_AUPR[runtimes]))
        print('mean_AUC:', mean_AUCs[runtimes], 'mean_AUPR:', mean_AUPRs[runtimes])
        runtimes = runtimes + 1
    MEAN_AUC = np.mean(np.array(mean_AUCs))
    MEAN_AUPR = np.mean(np.array(mean_AUPRs))
    print('MEAN_AUC:', MEAN_AUC, 'MEAN_AUPR:', MEAN_AUPR)
    fout = open(resultpath, 'a+')
    fout.write(mode+InPath + '####' + DFPath + '####' + TFPath + '####' + str(oversample) + '####' + str(Epoch) + '####' + str(learn_rate) + '####' + str(Batchsize) + '\n')
    fout.write('\n')
    fout.write('AUC:' + str(all_AUC) + '\n' + 'AUPR:' + str(all_AUPR) + '\n')
    fout.write('mean_AUCs:' + str(mean_AUCs) + '####' + 'mean_AUPRs:' + str(mean_AUPRs) + '\n')
    fout.write('MEAN_AUC:' + str(MEAN_AUC) + '####' + 'MEAN_AUPR:' + str(MEAN_AUPR) + '\n')
    fout.write('\n')
    fout.close()




