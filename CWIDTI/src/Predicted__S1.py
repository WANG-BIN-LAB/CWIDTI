import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve,f1_score,precision_score,recall_score,accuracy_score
import warnings
import sklearn.exceptions
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from imblearn.over_sampling import SMOTE
import scipy.io
from keras.callbacks import ModelCheckpoint
import pdb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
def loadDatadet(infile):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    return dataset
def n_k_fold(k, model,data,ntimes):
    mean_AUCs = [0 for _ in range(ntimes)]
    mean_AUPRs = [0 for _ in range(ntimes)]
    mean_F1s = [0 for _ in range(ntimes)]
    data=dataset
    all_acc = [[0] * k for _ in range(ntimes)]
    all_AUPR=[[0] * k for _ in range(ntimes)]
    all_AUC=[[0] * k for _ in range(ntimes)]
    all_F1=[[0] * k for _ in range(ntimes)]
    for a in range(0,ntimes):
      print('run #', a + 1)
      if (oversample == 1):
            model_smote = SMOTE()
            x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x_train, y_train)
            cv = StratifiedKFold(n_splits=k)
            i=0
            for train, test in cv.split(x_smote_resampled, y_smote_resampled):
                print('run #', a + 1, 'Processing fold #', i+1)
                earlystop=EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)
                filepath = dataset+'_weights.best.hdf5'
                checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

                history=model.fit(x_smote_resampled[train], y_smote_resampled[train],epochs=Epoch,batch_size=Batchsize,verbose=True,callbacks=[checkpoint, earlystop],validation_data=(x_smote_resampled[test],y_smote_resampled[test]))
                val_pre_pro = model.predict_proba(x_smote_resampled[test],batch_size=Batchsize,verbose=False)
                val_pre_class=model.predict_classes(x_smote_resampled[test],batch_size=Batchsize,verbose=False)
                fpr, tpr, auc_thresholds = roc_curve(y_smote_resampled[test], val_pre_pro,pos_label=1)
                auc_score = auc(fpr, tpr)
                precision, recall, pr_threshods = precision_recall_curve(y_smote_resampled[test], val_pre_pro,pos_label=1)
                aupr_score = auc(recall, precision)
                acc = accuracy_score(y_smote_resampled[test], val_pre_class)
                f1 = f1_score(y_smote_resampled[test], val_pre_class)
                pre = precision_score(y_smote_resampled[test], val_pre_class)
                sen = recall_score(y_smote_resampled[test], val_pre_class)

                pdb.set_trace()
                print('Data=', dataset)
                print('AUC=', auc_score)
                print('AUPR=', aupr_score)
                print(zxy)
                scores = model.evaluate(x_smote_resampled[test],y_smote_resampled[test], verbose=0)
                all_acc[a][i]=scores[1]
                all_AUPR[a][i]=aupr_score
                all_AUC[a][i]=auc_score
                all_F1[a][i]=f1
                i=i+1
                json_string = model.to_json()
                modelpath = "./" + data + "_transfer_architecture.json"
                weightpath = "./" + data+ "_transfer_weight.h5"
                open(modelpath , 'w').write(json_string)
                model.save_weights(weightpath)
      else:
          cv = StratifiedKFold(n_splits=10)
          i = 0
          for train, test in cv.split(x_train, y_train):
              print('run #', a + 1, 'Processing fold #', i + 1)
              earlystop = EarlyStopping(monitor='val_accuracy', verbose=1, min_delta=0.00001, patience=200)
              filepath = './zxy_start/result/' + dataset + '/model/' + dataset + '_CWI_' + '.hdf5'
              checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',
                                           period=1)
              history = model.fit(x_train[train], y_train[train], epochs=Epoch, batch_size=Batchsize,
                                  verbose=True, callbacks=[checkpoint, earlystop], validation_data=(x_train[test], y_train[test]))
              val_pre_pro = model.predict_proba(x_train[test], batch_size=Batchsize, verbose=False)
              val_pre_class = model.predict_classes(x_train[test], batch_size=Batchsize, verbose=False)
              fpr, tpr, auc_thresholds = roc_curve(y_smote_resampled[test], val_pre_pro, pos_label=1)
              auc_score = auc(fpr, tpr)
              precision, recall, pr_threshods = precision_recall_curve(y_smote_resampled[test], val_pre_pro,
                                                                       pos_label=1)
              aupr_score = auc(recall, precision)
              acc = accuracy_score(y_smote_resampled[test], val_pre_class)
              f1 = f1_score(y_smote_resampled[test], val_pre_class)
              pre = precision_score(y_smote_resampled[test], val_pre_class)
              sen = recall_score(y_smote_resampled[test], val_pre_class)
              pdb.set_trace()
              all_acc[a][i] = scores[1]
              all_AUPR[a][i] = aupr_score
              all_AUC[a][i] = auc_score
              all_F1[a][i] = f1
              i = i + 1
              json_string = model.to_json()
              modelpath = "./" + data + "_architecture.json"
              weightpath = "./" + data + "_weight.h5"
              open(modelpath, 'w').write(json_string)
              model.save_weights(weightpath)
      mean_AUCs[a]=np.mean(np.array(all_AUC[a]))
      mean_AUPRs[a]=np.mean(np.array(all_AUPR[a]))
      mean_F1s[a]=np.mean(np.array(all_F1[a]))
      print('mean_AUC:', mean_AUCs[a], 'mean_AUPR:', mean_AUPRs[a], 'mean_F1:', mean_F1s[a])
    return mean_AUCs,mean_AUPRs,mean_F1s,all_F1,all_AUPR,all_AUC
def build_model():
    model = Sequential()
    model.add(Dense(input_dim=(len(DT_feature[0])), output_dim=300, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=300, output_dim=200, init='he_normal'))#500
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=200, output_dim=100, init='he_normal'))#500
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=100, output_dim=1, init='he_normal'))  ##500
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    print(model.summary())
    return model
def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label
if __name__ == '__main__':
    dataset = "drugbank"
    Batchsize =1024
    Epoch = 200
    learn_rate = 0.001
    oversample = 1
    predict_num=100
    InPath = "./data/" + dataset +'/'+ "interaction.txt"
    DFPath = "./data3/" + dataset +'/'+dataset+ "_CWI_i.mat"
    TFPath = "./data3/" + dataset +'/'+dataset+ "_CWI_t.mat"
    data = np.loadtxt(InPath)
    data1 = np.reshape(data, (data.shape[0] * data.shape[1], 1))
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
    y_train = data1
    mean_AUCs, mean_AUPRs, mean_F1s, all_F1, all_AUPR, all_AUC = n_k_fold(5, build_model(), dataset,10)
    MEAN_AUC = np.mean(np.array(mean_AUCs))
    MEAN_AUPR = np.mean(np.array(mean_AUPRs))
    MEAN_F1 = np.mean(np.array(mean_F1s))
    print('MEAN_AUC:', MEAN_AUC, 'MEAN_AUPR:', MEAN_AUPR, 'MEAN_F1:', MEAN_F1)
    resultpath = "./result/"+dataset+'/'+dataset+"_CWM-DTI_S1.txt"
    fout = open(resultpath, 'a+')
    fout.write(InPath + '####' + DFPath + '####' + TFPath + '####' + str(oversample) + '####' + str(Epoch) + '####' + str(
            learn_rate) + '####' + str(Batchsize) + '\n')
    fout.write('\n')
    fout.write('AUC:' + str(all_AUC) + '\n' + 'AUPR:' + str(all_AUPR) + '\n' + 'F1:' + str(all_F1) + '\n')
    fout.write('mean_AUCs:' + str(mean_AUCs) + '####' + 'mean_AUPRs:' + str(mean_AUPRs) + '####' + 'mean_F1s:' + str(
        mean_F1s) + '\n')
    fout.write('MEAN_AUC:' + str(MEAN_AUC) + '####' + 'MEAN_AUPR:' + str(MEAN_AUPR) + '####' + 'MEAN_F1:' + str(
        MEAN_F1) + '\n')
    fout.write('\n')
    fout.close()
