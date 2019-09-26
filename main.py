import os
import numpy as np
import argparse
import pickle
import json
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

import Model as LM

print("Loading configure")
config=LM.get_config()

with open(config.pkl_file,"rb") as fp:
    data=pickle.load(fp)
fp.close()

X=data["array"]
y=data["cluster"]
features=data["genes"]
feature_to_key={key+1:gene for key,gene in enumerate(features)}
print("There are : {} features in data".format(len(features)))

print("Encoder cluster -> label ")
encoder=LabelEncoder()
encoder.fit(y)
label=encoder.transform(y)
print(Counter(label))

num_classes=len(np.unique(label))
weight=compute_class_weight("balanced",np.unique(label),label)
class_weight={i:w for i,w in enumerate(weight)}
print(class_weight)

X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=config.test_ratio,shuffle=True)
print("X train : ({},{})".format(X_train.shape[0],X_train.shape[1]))
print("X test : ({},{})".format(X_test.shape[0],X_test.shape[1]))

train_data = lgb.Dataset(data=X_train,label=y_train)
test_data = lgb.Dataset(data=X_test,label=y_test)

print("Build Model")
model=LM.Build_Model(config,class_weight)

print("training")
model.fit(X_train,y_train,
    eval_metric=["multi_logloss","multi_error"])

#print('Feature importances:', list(model.feature_importance()))
print('Saving model...')
# save model to file

print('Dumping and loading model with pickle...')
# dump model with pickle
with open(os.path.join(config.save_dir,'model.pkl'), 'wb') as fout:
    pickle.dump(model, fout)

