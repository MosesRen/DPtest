# -*- coding: utf-8 -*-
__author__ = "mosesren"
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score 
from PCAAAAI import pca as pcaaaai
from PCCN import pca as pcacn
#导入numpy包 pandas包 sklearn的数据集分割包

def load_traindata(filepath):
	#读取数据集

	csv = pd.read_csv(filepath,index_col = None,parse_dates = True,iterator = True)
	dataset = csv.read()
	# # 读取数据集
	return dataset
def OneHot(dataset):
	#对分类属性进行onehot编码

	datanum=dataset.iloc[:,[0,2,4,10,11,12]]
	workclass = pd.get_dummies(dataset['workclass'])
	race = pd.get_dummies(dataset['race'])
	sex = pd.get_dummies(dataset['sex'])
	result = datanum.join(workclass).join(race).join(sex)
	# result.to_csv(r'./dataset/onehottest.csv',index=False,sep=',')
	return result
def adult_removenull(filepath):
	#分别去除workclass，occupation，nativecountry中含有空值的行

	csv = pd.read_csv(filepath,index_col = None,parse_dates = True,iterator = True)
	dataset = csv.read()

	a = dataset[(dataset.workclass !=' ?')]
	b = a[(a.occupation !=' ?')]
	c = b[(b.nativecountry !=' ?')]

	# c.to_csv(r'./dataset/adlutNoNull.csv',index=False,sep=',')
	#输出处理后的文件
	return c
def codeCategory(dataset):
	for i in range(dataset.shape[1]):
		if dataset.iloc[0,i] == 'Category':
			categories= dataset.iloc[1:,i].astype("category").cat.categories
			for j in range(len(categories)):
				dataset.iloc[:,i] = dataset.iloc[:,i].replace(categories[j],j)
	return dataset
#	return dataset0
if __name__ == '__main__':
	dataset = load_traindata(r'../dataset/adlutNoNull.csv')
	# train_data = OneHot(dataset).values
	dataset = codeCategory(dataset)
	print(dataset)
	dataset.to_csv(r'../dataset/adlutcode.csv',index=False,sep=',')



	# train_data = dataset.iloc[1:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values
	# # #onehot编码
	# train_target = dataset.loc[1:,['class']].values
	# # # # 读取标签列
	# train_data = normalize(train_data)
	# # #数据归一化
	# train_data,recov= pcacn(train_data, 12)
	# # #主成分分析
	# X_train, X_test, y_train, y_test =  train_test_split(train_data, train_target,test_size=0.3, random_state=42)
	# # 分割训练集测试集
	# # from sklearn.svm import SVC
	# # #SVM训练
	# # svm_model = SVC(gamma='auto')
	# # svm_fit_model = svm_model.fit(X_train,y_train.ravel())
	# # y_pred_svm = svm_model.predict(X_test)
	# # y_true_svm = y_test.ravel()
	# # print('Accuracy = ', accuracy_score(y_true_svm, y_pred_svm))

	# from sklearn.ensemble import RandomForestClassifier
	# #随机森林训练
	# rf_model =  RandomForestClassifier(n_estimators=15, n_jobs=8)
	# rf_fit_model = rf_model.fit(X_train, y_train.ravel())
	# y_pred_rf = rf_model.predict(X_test)
	# y_true_rf = y_test.ravel()
	# print('Accuracy = ', accuracy_score(y_true_rf, y_pred_rf))
