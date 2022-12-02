import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def train_meta(classifiers_map,train_features,train_re_labels,base_classifiers,test_features,test_labels):
    ls = []
    for i in range(len(base_classifiers)):
        ls.append((classifiers_map[base_classifiers[i]].predict_proba(train_features))[:,1])
    new_train_features = np.array(ls).transpose()
    clf_meta_modellog = LogisticRegression()
    clf_meta_modellog.fit(new_train_features,train_re_labels)
    clf_meta_modelnb = GaussianNB()
    clf_meta_modelnb.fit(new_train_features,train_re_labels)
    clf_meta_model_AdaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    clf_meta_model_AdaBoost.fit(new_train_features,train_re_labels)
    ls = []
    for i in range(len(base_classifiers)):
        ls.append((classifiers_map[base_classifiers[i]].predict_proba(test_features))[:,1])
    new_test_features = np.array(ls).transpose()
    final_list = list((1/3)*(clf_meta_modellog.predict(new_test_features))+(1/3)*(clf_meta_modelnb.predict(new_test_features))+(1/3)*(clf_meta_model_AdaBoost.predict(new_test_features)))
    for i in range(len(final_list)):
        final_list[i]=round(final_list[i])
    # print(final_list)
    return final_list

    # +(1/3)*(clf_meta_model_AdaBoost.predict(new_test_features))
