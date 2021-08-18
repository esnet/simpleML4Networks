#DecisionTreeClassifier 

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import os

import pandas as pd 
import numpy as np
from sklearn import cluster, datasets

import graphviz 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from sklearn import metrics


cfile_normal_data = os.path.join("datasets/tstatDT/normal.csv")
cdfnormaldata=pd.read_csv(cfile_normal_data)
cfile_loss1_data = os.path.join("datasets/tstatDT/packetloss1.csv")
cdfloss1data=pd.read_csv(cfile_loss1_data)

cfile_loss3_data = os.path.join("datasets/tstatDT/packetloss3.csv")
cdfloss3data=pd.read_csv(cfile_loss3_data)

cfile_loss5_data = os.path.join("datasets/tstatDT/packetloss5.csv")
cdfloss5data=pd.read_csv(cfile_loss5_data)



cdfnormaldata['Target'] = 0
cdfloss1data['Target'] = 1
cdfloss3data['Target'] = 2
cdfloss5data['Target'] = 3



all_data_frames = [cdfnormaldata, cdfloss1data, cdfloss3data, cdfloss5data]
all_data_result = pd.concat(all_data_frames)

#convert all strings and time to zero- to remove them from rules
all_data_result['#15#c_ip:1']=0
all_data_result['s_ip:15']=0
all_data_result['first:29']=0	
all_data_result['last:30']=0
all_data_result['http_res:113']=0
all_data_result['c_tls_SNI:116']=0
all_data_result['s_tls_SCN:117']=0
all_data_result['fqdn:127']=0	
all_data_result['dns_rslv:128']=0
all_data_result['http_hostname:131']=0
all_data_result['c_port:2']=0
all_data_result['s_port:16']=0
all_data_result['first:29']=0
all_data_result['last:30']=0

features=list(all_data_result.columns[:133])

print(all_data_result['http_res:113'])

X=all_data_result[features]
y=all_data_result['Target']
target_names=['0','1','2','3']


#Original tree

clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random',max_leaf_nodes=30, min_samples_leaf = 100, max_depth= 9)
clf = clf.fit(X, y)




print("Train score =")
print(clf.score(X,y))
#pruning for optimum parameters



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

y_pred_1 = clf.predict(X_test)
print("Accuracy on full tree:",metrics.accuracy_score(y_test, y_pred_1))

# first fit the model to get baseline
#clf=clf.fit(X_test, y_test)
print("test tree score =")
print(clf.score(X_test, y_test))


#evaluating optimum parameters
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy of test tree:",metrics.accuracy_score(y_test, y_pred))
#printing AUC
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#roc_auc = auc(false_positive_rate, true_positive_rate)

#print("roc_auc =")
#print(roc_auc)

#tuning max depth

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
	dt = tree.DecisionTreeClassifier(max_depth=max_depth)
	dt.fit(X_train, y_train)
	train_pred = dt.predict(X_train)

	tscore=metrics.accuracy_score(y_train, train_pred)
	train_results.append(tscore)

	y_pred=dt.predict(X_test)
	###
	
	nscore=metrics.accuracy_score(y_test, y_pred)
	test_results.append(nscore)

#print(train_results)
line1, = plt.plot(max_depths, train_results, 'b', label="Training Data")
line2, = plt.plot(max_depths, test_results, 'r', label="Test Data")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score')
plt.xlabel('Tree depth')
plt.show()


train_results2 = []
test_results2 = []
#tuning the leaf nodes
for j in [10,30,50, 60,70, 80, 90, 100, 500, 1000]:
	clf_stump=tree.DecisionTreeClassifier(max_features=None,max_leaf_nodes=j)
	clf_stump.fit(X_train, y_train)
	train_pred2 = clf_stump.predict(X_train)
	tscore2=metrics.accuracy_score(y_train, train_pred2)
	train_results2.append(tscore2)

	newy_pred=clf_stump.predict(X_test)

	nscore2=metrics.accuracy_score(y_test, newy_pred)
	test_results2.append(nscore2)
	print(j,tscore2, nscore2)


  


#clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 20, min_samples_leaf = 10, max_depth= 10)
#clf = clf.fit(X, y)
clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random',max_leaf_nodes=30, min_samples_leaf = 100, max_depth= 9)
clf = clf.fit(X, y)
print("Tree has been generated, Saving to PDF")
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("hamilton-loss")

dot_data = tree.export_graphviz(clf, out_file=None, 


  filled=True, rounded=True,  


  special_characters=True,


  feature_names=features,  


  class_names=['0','1','2','3'])  
graph = graphviz.Source(dot_data)  
graph


feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("Print top Features: feat importance = " + str(feat_importance))


