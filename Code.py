#%% load package
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import miceforest as mf
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from matplotlib import pyplot
from numpy import argmax
from functools import reduce
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from io import StringIO
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import pickle
import sklearn
import json
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import shap
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
%matplotlib
sns.set()

#%% load data
train_data = pd.read_csv('F:\\R语言\\2022_11_14code_R\\code\\zhi_pred_model_ml\\train.txt',sep="\t",index_col=(0))
test_data = pd.read_csv('F:\\R语言\\2022_11_14code_R\\code\\zhi_pred_model_ml\\test.txt',sep="\t",index_col=(0))

#%% Variable filtering
features = ["Primary_Site","Marital_status","Laterality","Tumor_size",'Radiation',"Chemotherapy",'T']
indicator = ["N"]
X_train = train_data[features] 
X_test = test_data[features]
y_train = train_data[indicator]
y_test = test_data[indicator]
y_train = label_binarize(y_train, classes=[1,2])
y_test = label_binarize(y_test, classes=[1,2])
X_train = np.array(X_train)
X_test = np.array(X_test)
X = X_train
y = y_train

#%% Data imbalance processing

random_state_new = 50
oversample = SMOTE(random_state = random_state_new)
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_test, y_test = oversample.fit_resample(X_test, y_test)

#%% Build machine learning models
jc = 10
lr = LogisticRegression(penalty="none", random_state=random_state_new)
rf = RandomForestClassifier(
    n_estimators=200,  max_features=4, random_state=random_state_new)
dt = tree.DecisionTreeClassifier(
    min_weight_fraction_leaf=0.25, random_state=random_state_new)
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
xgb_model = xgb.XGBClassifier(
    n_estimators=360, max_depth=2, learning_rate=1, random_state=random_state_new)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)

logis_model = LogisticRegression(random_state=random_state_new,
                                 solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
logis_model.score(X_test, y_test)
KNN_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
KNN_model.score(X_test, y_test)
GaNB_model = GaussianNB().fit(X_train, y_train)
GaNB_model.score(X_test, y_test)
tree = tree.DecisionTreeClassifier(random_state=random_state_new)
tree_model = tree.fit(X_train, y_train)
tree_model.score(X_test, y_test)
Bag = BaggingClassifier(KNeighborsClassifier(
), max_samples=0.5, max_features=0.5, random_state=random_state_new)
Bag_model = Bag.fit(X_train, y_train)
Bag_model.score(X_test, y_test)
RF = RandomForestClassifier(n_estimators=10, max_depth=3,
                            min_samples_split=12, random_state=random_state_new)
RF_model = RF.fit(X_train, y_train)
RF_model.score(X_test, y_test)
ET = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=2, random_state=random_state_new)
ET_model = ET.fit(X_train, y_train)
ET_model.score(X_test, y_test)
AB = AdaBoostClassifier(n_estimators=10, random_state=random_state_new)
AB_model = AB.fit(X_train, y_train)
AB_model.score(X_test, y_test)
GBT = GradientBoostingClassifier(
    n_estimators=10, learning_rate=1.0, max_depth=1, random_state=random_state_new)
GBT_model = GBT.fit(X_train, y_train)
GBT_model.score(X_test, y_test)
clf1 = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=random_state_new)
clf3 = GaussianNB()
VOTE = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
VOTE_model = VOTE.fit(X_train, y_train)
VOTE_model.score(X_test, y_test)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
gbm_model = gbm.fit(X_train, y_train)
gbm_model.score(X_test, y_test)
xgb_model = xgb.XGBClassifier(
    n_estimators=360, max_depth=2, learning_rate=1, random_state=random_state_new)
xgb_model = xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)

#%% Internal cross-validate line plots
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
plt.plot(x, aucs_AB_model, label='AB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_AB_model).mean(), np.array(aucs_AB_model).std()),
         linewidth=3, color='#03a8f3', marker='>', markerfacecolor='#03a8f3', markersize=12)
plt.plot(x, aucs_logis_model, label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_logis_model).mean(), np.array(aucs_logis_model).std()),
         linewidth=3, color='#fe5722', marker='>', markerfacecolor='#fe5722', markersize=12)
plt.plot(x, aucs_mlp_model, label='MLP: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_mlp_model).mean(), np.array(aucs_mlp_model).std()),
         linewidth=3, color='#009587', marker='>', markerfacecolor='#009587', markersize=12)
plt.plot(x, aucs_Bag_model, label='BAG: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_Bag_model).mean(), np.array(aucs_Bag_model).std()),
         linewidth=3, color='#673ab6', marker='>', markerfacecolor='#673ab6', markersize=12)
plt.plot(x, aucs_gbm_model, label='GBM: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_gbm_model).mean(), np.array(aucs_gbm_model).std()),
         linewidth=3, color='#b5da3d', marker='>', markerfacecolor='#b5da3d', markersize=12)
plt.plot(x, aucs_xgb_model, label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_xgb_model).mean(), np.array(aucs_xgb_model).std()),
         linewidth=3, color='#3f51b4', marker='>', markerfacecolor='#3f51b4', markersize=12)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim(0.55, 0.92)
plt.xlim(0.7, 10.3)
plt.xlabel('Round of Cross')
plt.ylabel('AUC')
plt.title('Ten Fold Cross Validation')
plt.legend(loc=4)
plt.show()

#%% ROC curve
random_state_new = random_state_new
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=random_state_new)
AB_model.fit(X_train, y_train)
logis_model.fit(X_train, y_train)
mlp.fit(X_train, y_train)
Bag_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
fpr_ab, tpr_ab, thresholds_ab = roc_curve(
    y_test, AB_model.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(
    y_test, logis_model.predict_proba(X_test)[:, 1])
fpr_bag, tpr_bag, thresholds_bag = roc_curve(
    y_test, Bag_model.predict_proba(X_test)[:, 1])
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(
    y_test, mlp.predict_proba(X_test)[:, 1])
fpr_gbm, tpr_gbm, thresholds_gbm = roc_curve(
    y_test, gbm_model.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(
    y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc_ab = auc(fpr_ab, tpr_ab)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_bag = auc(fpr_bag, tpr_bag)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
plt.plot(fpr_lr, tpr_lr, label="ROC Curve LR; AUC = {:.3f}".format(roc_auc_lr))
plt.plot(fpr_ab, tpr_ab, label="ROC Curve AB; AUC = {:.3f}".format(roc_auc_ab))
plt.plot(fpr_bag, tpr_bag,
          label="ROC Curve BAG; AUC = {:.3f}".format(roc_auc_bag))
plt.plot(fpr_mlp, tpr_mlp,
          label="ROC Curve MLP; AUC = {:.3f}".format(roc_auc_mlp))
plt.plot(fpr_gbm, tpr_gbm,
          label="ROC Curve GBM; AUC = {:.3f}".format(roc_auc_gbm))
plt.plot(fpr_xgb, tpr_xgb,
          label="ROC Curve XGB; AUC = {:.3f}".format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='Reference')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.show()
plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\NOS_WSN\\ROC curve.pdf')

#%% Index calculation
# f = ab
rf_score = AB_model.score(X_test, y_test)
lr_score = logis_model.score(X_test, y_test)
#bag = dt
dt_score = Bag_model.score(X_test, y_test)
mlp_score = mlp.score(X_test, y_test)
gbm_score = gbm_model.score(X_test, y_test)
xgb_score = xgb_model.score(X_test, y_test)
rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
# 混淆矩阵
rf_cf = confusion_matrix(y_test, rf_prob)
lr_cf = confusion_matrix(y_test, lr_prob)
dt_cf = confusion_matrix(y_test, dt_prob)
mlp_cf = confusion_matrix(y_test, mlp_prob)
gbm_cf = confusion_matrix(y_test, gbm_prob)
xgb_cf = confusion_matrix(y_test, xgb_prob)
rf_cf
lr_cf
dt_cf
mlp_cf
gbm_cf
xgb_cf
TN_rf, FP_rf, FN_rf, TP_rf = confusion_matrix(y_test, rf_prob).ravel()
TN_lr, FP_lr, FN_lr, TP_lr = confusion_matrix(y_test, lr_prob).ravel()
TN_dt, FP_dt, FN_dt, TP_dt = confusion_matrix(y_test, dt_prob).ravel()
TN_mlp, FP_mlp, FN_mlp, TP_mlp = confusion_matrix(y_test, mlp_prob).ravel()
TN_gbm, FP_gbm, FN_gbm, TP_gbm = confusion_matrix(y_test, gbm_prob).ravel()
TN_xgb, FP_xgb, FN_xgb, TP_xgb = confusion_matrix(y_test, xgb_prob).ravel()
sen_rf, spc_rf = round(TP_rf/(TP_rf+FN_rf), 3), round(TN_rf/(FP_rf+TN_rf), 3)
sen_lr, spc_lr = round(TP_lr/(TP_lr+FN_lr), 3), round(TN_lr/(FP_lr+TN_lr), 3)
sen_dt, spc_dt = round(TP_dt/(TP_dt+FN_dt), 3), round(TN_dt/(FP_dt+TN_dt), 3)
sen_mlp, spc_mlp = round(TP_mlp/(TP_mlp+FN_mlp),
                          3), round(TN_mlp/(FP_mlp+TN_mlp), 3)
sen_gbm, spc_gbm = round(TP_gbm/(TP_gbm+FN_gbm),
                          3), round(TN_gbm/(FP_gbm+TN_gbm), 3)
sen_xgb, spc_xgb = round(TP_xgb/(TP_xgb+FN_xgb),
                          3), round(TN_xgb/(FP_xgb+TN_xgb), 3)
AB_f1 = f1_score(y_test, rf_prob, average='macro')
LR_f1 = f1_score(y_test, lr_prob, average='macro')
DT_f1 = f1_score(y_test, dt_prob, average='macro')
MLP_f1 = f1_score(y_test, mlp_prob, average='macro')
GBM_f1 = f1_score(y_test, gbm_prob, average='macro')
XGB_f1 = f1_score(y_test, xgb_prob, average='macro')

print("AB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(AB_f1, roc_auc_ab, rf_score, sen_rf, spc_rf))
print("LR的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(LR_f1, roc_auc_lr, lr_score, sen_lr, spc_lr))
print("BAG的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(DT_f1, roc_auc_bag, dt_score, sen_dt, spc_dt))
print("MLP的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(MLP_f1, roc_auc_mlp, mlp_score, sen_mlp, spc_mlp))
print("GBM的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(GBM_f1, roc_auc_gbm, gbm_score, sen_gbm, spc_gbm))
print("XGB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(XGB_f1, roc_auc_xgb, xgb_score, sen_xgb, spc_xgb))

#%% Confusion matrix rendering
XGB_prob1 = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, XGB_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of XGB")
plt.show()

plt.subplot(2,3,2)
AB_prob1 = AB_model.predict(X_test)
cm = confusion_matrix(y_test, AB_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of AB")
plt.show()

plt.subplot(2,3,3)
LR_prob1 = logis_model.predict(X_test)
cm = confusion_matrix(y_test, LR_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of LR")

plt.subplot(2,3,4)
GBM_prob1 = gbm_model.predict(X_test)
cm = confusion_matrix(y_test, GBM_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of GBM")

plt.subplot(2,3,5)
MLP_prob1 = mlp.predict(X_test)
cm = confusion_matrix(y_test, MLP_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of MLP")

plt.subplot(2,3,6)
BAG_prob1 = Bag_model.predict(X_test)
cm = confusion_matrix(y_test, BAG_prob1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None Metastases', 'Metastases'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of BAG")
plt.show()



#%% Calibration curve
rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
plt.rcParams["axes.grid"] = False
sns.set()
from sklearn.calibration import calibration_curve
def calibration_curve_1(k,y_pred,y_true,method_name,color,title):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=5)
    plt.figure(k)
    plt.plot(prob_pred,prob_true,color=color,label='%s calibration_curve'%method_name,marker='s')
    plt.plot([i/100 for i in range(0,100)],[i/100 for i in range(0,100)],color='black',linestyle='--')
    plt.xlim(-0.02,1.02,0.2)
    plt.ylim(-0.02,1.02,0.2)
    plt.xlabel('y_preds')
    plt.ylabel('y_real')
    plt.title(title)
    plt.legend(loc='lower right')

plt.subplot(1,2,1)
calibration_curve_1(1,logis_model.predict_proba(X_train)[:,1],y_train,'LR','red','train calibration_curve')
calibration_curve_1(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','train calibration_curve')
calibration_curve_1(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','train calibration_curve')
calibration_curve_1(1,mlp.predict_proba(X_train)[:,1],y_train,'MLP','green','train calibration_curve')
calibration_curve_1(1,gbm_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','train calibration_curve')
calibration_curve_1(1,xgb_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','train calibration_curve')

plt.subplot(1,2,2)
calibration_curve_1(1,logis_model.predict_proba(X_test)[:,1],y_test,'LR','red','test calibration_curve')
calibration_curve_1(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','test calibration_curve')
calibration_curve_1(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','test calibration_curve')
calibration_curve_1(1,mlp.predict_proba(X_test)[:,1],y_test,'MLP','green','test calibration_curve')
calibration_curve_1(1,gbm_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','test calibration_curve')
calibration_curve_1(1,xgb_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','test calibration_curve')
plt.show()




#%% ROC curve of Train and Test
plt.style.use('tableau-colorblind10')
def plot_roc(k,y_pred_undersample_score,labels_test,classifiers,color,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels_test.values.ravel(),y_pred_undersample_score)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure(k)
    plt.title(title)
    plt.plot(fpr, tpr, 'b',color=color,label='%s AUC = %0.3f'% (classifiers,roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.02,1.0])
    plt.ylim([-0.02,1.01])
    plt.xlabel("1 - Specifity")
    plt.ylabel("Sensitivity")
    
sns.set()
plt.subplot(1,2,1)
plot_roc(1,logis_model.predict_proba(train[f_top])[:,1],train['label'],'LR','red','ROC curve of Train set')
plot_roc(1,AB_model.predict_proba(train[f_top])[:,1],train['label'],'AB','blue','ROC curve of Train set')
plot_roc(1,Bag_model.predict_proba(train[f_top])[:,1],train['label'],'BAG','m','ROC curve of Train set')
plot_roc(1,mlp.predict_proba(train[f_top])[:,1],train['label'],'MLP','green','ROC curve of Train set')
plot_roc(1,gbm_model.predict_proba(train[f_top])[:,1],train['label'],'GBM','tomato','ROC curve of Train set')
plot_roc(1,xgb_model.predict_proba(train[f_top])[:,1],train['label'],'XGB','darkblue','ROC curve of Train set')

plt.subplot(1,2,2)
plot_roc(1,logis_model.predict_proba(test[f_top])[:,1],test['label'],'LR','red','ROC curve of Test set')
plot_roc(1,AB_model.predict_proba(test[f_top])[:,1],test['label'],'AB','blue','ROC curve of Test set')
plot_roc(1,Bag_model.predict_proba(test[f_top])[:,1],test['label'],'BAG','m','ROC curve of Test set')
plot_roc(1,mlp.predict_proba(test[f_top])[:,1],test['label'],'MLP','green','ROC curve of Test set')
plot_roc(1,gbm_model.predict_proba(test[f_top])[:,1],test['label'],'GBM','tomato','ROC curve of Test set')
plot_roc(1,xgb_model.predict_proba(test[f_top])[:,1],test['label'],'XGB','darkblue','ROC curve of Test set')
plt.show()


#%%PR Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
def ro_curve(k,y_pred, y_label, method_name,color,title):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)    
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
    plt.figure(k)
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.3f)' % average_precision_score(y_label, y_pred),color=color)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title(title)
    plt.legend(loc='upper right')
fig = plt.gcf()
#train 
plt.subplot(1,2,1)
ro_curve(1,logis_model.predict_proba(X_train)[:,1],y_train,'LR','red','train Precision Recall Curve')
ro_curve(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','train Precision Recall Curve')
ro_curve(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','train Precision Recall Curve')
ro_curve(1,mlp.predict_proba(X_train)[:,1],y_train,'MLP','green','train Precision Recall Curve')
ro_curve(1,gbm_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','train Precision Recall Curve')
ro_curve(1,xgb_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','train Precision Recall Curve')

#test
plt.subplot(1,2,2)
ro_curve(1,logis_model.predict_proba(X_test)[:,1],y_test,'LR','red','test Precision Recall Curve')
ro_curve(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','test Precision Recall Curve')
ro_curve(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','test Precision Recall Curve')
ro_curve(1,mlp.predict_proba(X_test)[:,1],y_test,'MLP','green','test Precision Recall Curve')
ro_curve(1,gbm_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','test Precision Recall Curve')
ro_curve(1,xgb_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','test Precision Recall Curve')
plt.show()

#%% DCA curve
from sklearn import preprocessing
def dac(pred_ans,train,f_top,k,color,name,title,aaa=0.05):
    Y = train['label']
    a=train['label'].value_counts()[0]
    b=train['label'].value_counts()[1]
    pt_arr = []
    net_bnf_arr = []
    jiduan = []
    pred_ans = pred_ans.ravel()
    for i in range(0,100,1):
        pt = i /100
        pred_ans_clip = np.zeros(pred_ans.shape[0])
        for j in range(pred_ans.shape[0]):
            if pred_ans[j] >= pt:
                pred_ans_clip[j] = 1
            else:
                pred_ans_clip[j] = 0
        TP = np.sum((Y) * np.round(pred_ans_clip))
        FP = np.sum((1 - Y) * np.round(pred_ans_clip))
        net_bnf = ( TP-(FP * pt/(1-pt)) )/Y.shape[0]
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
        jiduan.append((b-a*pt/(1-pt))/(a+b))
    plt.figure(figsize=(12,8))
    plt.figure(k)
    plt.plot(pt_arr, net_bnf_arr, color=color, lw=2,label=name)
    plt.legend(loc=4)
    plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=2)
    pt_np = np.array(pt_arr)
    plt.plot(pt_arr, jiduan , color='b', lw=2, linestyle='dotted')
    plt.xlim([0.0, 1.0])
    plt.ylim([-1, aaa])
    plt.xlabel('Risk Threshold')
    plt.ylabel('Net Benefit')
    plt.title(title)

fig = plt.gcf()
plt.subplot(1,2,1)
dac(logis_model.predict_proba(train[f_top])[:,1],train,f_top,1,'red','LR','Train data')
dac(AB_model.predict_proba(train[f_top])[:,1],train,f_top,1,'blue','AB','Train data')
dac(Bag_model.predict_proba(train[f_top])[:,1],train,f_top,1,'m','BAG','Train data')
dac(mlp.predict_proba(train[f_top])[:,1],train,f_top,1,'green','MLP','Train data')
dac(gbm_model.predict_proba(train[f_top])[:,1],train,f_top,1,'tomato','GBM','Train data')
dac(xgb_model.predict_proba(train[f_top])[:,1],train,f_top,1,'darkblue','XGB','Train data')

plt.subplot(1,2,2)
dac(logis_model.predict_proba(test[f_top])[:,1],test,f_top,1,'red','LR','Test data')
dac(AB_model.predict_proba(test[f_top])[:,1],test,f_top,1,'blue','AB','Test data')
dac(Bag_model.predict_proba(test[f_top])[:,1],test,f_top,1,'m','BAG','Test data')
dac(mlp.predict_proba(test[f_top])[:,1],test,f_top,1,'green','MLP','Test data')
dac(gbm_model.predict_proba(test[f_top])[:,1],test,f_top,1,'tomato','GBM','Test data')
dac(xgb_model.predict_proba(test[f_top])[:,1],test,f_top,1,'darkblue','XGB','Test data')
plt.show()

#%% SHAP
sns.set()
explainer = shap.Explainer(gbm_model, train_data[features])
shap_values = explainer.shap_values(train_data[features])  
a = 2862
shap.initjs()
plot1 = shap.force_plot(explainer.expected_value,
                shap_values[a, :], 
                train_data[features].iloc[a, :], 
                figsize=(15, 5),
                matplotlib=True,
                out_names = "Output value")

sns.set()
shap.summary_plot(shap_values, 
                  train_data[features],
                  plot_type="violin", 
                  max_display=7,
                  color='#fee494',
                  title='Feature importance')

shap.summary_plot(shap_values, train_data[features], plot_type="bar")

#%%Table 3
print("AB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(AB_f1, roc_auc_ab, rf_score, sen_rf, spc_rf))
print("LR的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(LR_f1, roc_auc_lr, lr_score, sen_lr, spc_lr))
print("BAG的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(DT_f1, roc_auc_bag, dt_score, sen_dt, spc_dt))
print("MLP的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(MLP_f1, roc_auc_mlp, mlp_score, sen_mlp, spc_mlp))
print("GBM的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(GBM_f1, roc_auc_gbm, gbm_score, sen_gbm, spc_gbm))
print("XGB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(XGB_f1, roc_auc_xgb, xgb_score, sen_xgb, spc_xgb))

#%% Saving output probability
prob = pd.DataFrame(mlp.predict_proba(np.array(X_data)))
prob.to_csv("E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\NM_WSN\\BAG_pro.csv", index=False, sep=',')
