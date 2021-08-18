#RandomForestTreeClassifier 

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import ShuffleSplit


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import pandas as pd 
import numpy as np
from sklearn import cluster, datasets

import graphviz 

from sklearn.tree import export_graphviz
import pydot

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from sklearn import metrics
from subprocess import call
from IPython.display import Image


efile_normal_datac = os.path.join("datasets/ele_nm.csv")
edfnormaldatac=pd.read_csv(efile_normal_datac)#nrows=250



efile_loss10_data = os.path.join("datasets/ele_loss10.csv")
edfloss10data=pd.read_csv(efile_loss10_data)



cfile_test_data = os.path.join("datasets/ele_nm_impure.csv")
cdftestdata=pd.read_csv(cfile_test_data)
cdftestdata.columns=['#15#c_ip:1','c_port:2','c_pkts_all:3','c_rst_cnt:4','c_ack_cnt:5',
'c_ack_cnt_p:6','c_bytes_uniq:7','c_pkts_data:8','c_bytes_all:9','c_pkts_retx:10',
'c_bytes_retx:11','c_pkts_ooo:12','c_syn_cnt:13','c_fin_cnt:14','s_ip:15',
's_port:16','s_pkts_all:17','s_rst_cnt:18','s_ack_cnt:19','s_ack_cnt_p:20',
's_bytes_uniq:21','s_pkts_data:22','s_bytes_all:23','s_pkts_retx:24',
's_bytes_retx:25','s_pkts_ooo:26','s_syn_cnt:27','s_fin_cnt:28','first:29',
'last:30','durat:31','c_first:32','s_first:33','c_last:34','s_last:35',
'c_first_ack:36','s_first_ack:37','c_isint:38','s_isint:39','c_iscrypto:40',
's_iscrypto:41','con_t:42','p2p_t:43','http_t:44','c_rtt_avg:45','c_rtt_min:46',
'c_rtt_max:47','c_rtt_std:48','c_rtt_cnt:49','c_ttl_min:50','c_ttl_max:51',
's_rtt_avg:52','s_rtt_min:53','s_rtt_max:54','s_rtt_std:55','s_rtt_cnt:56',
's_ttl_min:57','s_ttl_max:58','p2p_st:59','ed2k_data:60','ed2k_sig:61',
'ed2k_c2s:62','ed2k_c2c:63','ed2k_chat:64','c_f1323_opt:65','c_tm_opt:66',
'c_win_scl:67','c_sack_opt:68','c_sack_cnt:69','c_mss:70','c_mss_max:71',
'c_mss_min:72','c_win_max:73','c_win_min:74','c_win_0:75','c_cwin_max:76',
'c_cwin_min:77','c_cwin_ini:78','c_pkts_rto:79','c_pkts_fs:80','c_pkts_reor:81',
'c_pkts_dup:82','c_pkts_unk:83','c_pkts_fc:84','c_pkts_unrto:85','c_pkts_unfs:86',
'c_syn_retx:87','s_f1323_opt:88','s_tm_opt:89','s_win_scl:90','s_sack_opt:91',
's_sack_cnt:92','s_mss:93','s_mss_max:94','s_mss_min:95','s_win_max:96',
's_win_min:97','s_win_0:98','s_cwin_max:99','s_cwin_min:100','s_cwin_ini:101',
's_pkts_rto:102','s_pkts_fs:103','s_pkts_reor:104','s_pkts_dup:105',
's_pkts_unk:106','s_pkts_fc:107','s_pkts_unrto:108','s_pkts_unfs:109',
's_syn_retx:110','http_req_cnt:111','http_res_cnt:112','http_res:113',
'c_pkts_push:114','s_pkts_push:115','c_tls_SNI:116','s_tls_SCN:117',
'c_npnalpn:118','s_npnalpn:119','c_tls_sesid:120','c_last_handshakeT:121',
's_last_handshakeT:122','c_appdataT:123','s_appdataT:124','c_appdataB:125',
's_appdataB:126','fqdn:127','dns_rslv:128','req_tm:129','res_tm:130',
'DeclaredThroughput','Actualthroughput']
#print(cdftestdata)

edfnormaldatac['Target'] = 0
edfloss10data['Target'] = 1


#all_data_frames = [edfnormaldatac, mdfnormaldatac, edfnormaldatar, mdfnormaldatar,edfnormaldata, mdfnormaldata,edfnormaldatab, mdfnormaldatab]

all_data_frames = [edfnormaldatac, edfloss10data]


 #cdfloss1data, hamiltondfnormaldata, hamiltondfloss1data, 
#hamiltondfnormaldata,hamiltondfloss1data, BBRdfnormaldata,BBRdfloss1data]
all_data_result = pd.concat(all_data_frames)

test_data_frames = cdftestdata


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
all_data_result['http_t:44']=0


features=list(all_data_result.columns[:133])

test_data_frames['#15#c_ip:1']=0
test_data_frames['s_ip:15']=0
test_data_frames['first:29']=0	
test_data_frames['last:30']=0
test_data_frames['http_res:113']=0
test_data_frames['c_tls_SNI:116']=0
test_data_frames['s_tls_SCN:117']=0
test_data_frames['fqdn:127']=0	
test_data_frames['dns_rslv:128']=0
test_data_frames['http_hostname:131']=0
test_data_frames['c_port:2']=0
test_data_frames['s_port:16']=0
test_data_frames['first:29']=0
test_data_frames['last:30']=0
test_data_frames['http_t:44']=0


testfeatures=list(test_data_frames.columns[:133])


#print(all_data_result['http_res:113'])

X=all_data_result[features]
y=all_data_result['Target']
target_names=['0','1']
print(X)
testX=test_data_frames[testfeatures]
print(testX)

#Original tree
#X,y= make_classification(n_samples=1000, n_features=4,
#	n_informative=2, n_redundant=0,
#	random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)


clf = clf.fit(X, y)




print("train score =")
print(clf.score(X,y))
#pruning for optimum parameters



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

y_pred_1 = clf.predict(X_test)
print(y_pred_1)

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
	dt = RandomForestClassifier(max_depth=max_depth, max_features=15)
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


print("Printing leaf nodes?")
train_results2 = []
test_results2 = []
#tuning the leaf nodes
for j in [10,30,50, 60,70, 80, 90, 100]:
	clf_stump=RandomForestClassifier(max_depth=5,n_estimators=j,max_features=15)
	clf_stump.fit(X_train, y_train)
	train_pred2 = clf_stump.predict(X_train)
	tscore2=metrics.accuracy_score(y_train, train_pred2)
	train_results2.append(tscore2)

	newy_pred=clf_stump.predict(X_test)

	nscore2=metrics.accuracy_score(y_test, newy_pred)
	test_results2.append(nscore2)
	print(j,tscore2, nscore2)


#predicting values
#after parameter tuning




#clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 20, min_samples_leaf = 10, max_depth= 10)
#clf = clf.fit(X, y)
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0,max_features=15)
clf = clf.fit(X, y)
print("tree generated")
importances=clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

estimator=clf.estimators_[5]

#for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
 #   X_train, X_test = X[train_idx], X[test_idx]
  #  Y_train, Y_test = Y[train_idx], Y[test_idx]
  #  r = clf.fit(X_train, Y_train)
  #  acc = r2_score(Y_test, clf.predict(X_test))
  #  for i in range(X.shape[1]):
   #     X_t = X_test.copy()
   #     np.random.shuffle(X_t[:, i])
   #     shuff_acc = r2_score(Y_test, clf.predict(X_t))
   #     scores[names[i]].append((acc-shuff_acc)/acc)
#print("Features sorted by their score:")
#print(([(round(np.mean(score), 4), feat) for
 #             feat, score in scores.items()]))




clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0,max_features=15)
clf = clf.fit(X, y)


y_pred_1 = clf.predict(testX)
print("Test size %s" %len(testX))
for n in y_pred_1:
	print(n)
