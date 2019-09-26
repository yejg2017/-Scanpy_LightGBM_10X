import os
import numpy as np
import lightgbm as lgb
import time
import pickle
import argparse

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.utils.class_weight import compute_class_weight


def get_config():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--pkl_file",type=str,
            default="./log/array.pkl",
            help="the path of pickle file")

    parser.add_argument("--test_ratio",type=float,
            default=0.2,
            help="the ratio of test size")

    parser.add_argument("--n_jobs",type=int,
            default=4,
            help="the number threads for training")

    parser.add_argument("--learning_rate",type=float,
            default=0.1,
            help="the learning ratio for training")

    parser.add_argument("--boosting_type",type=str,
            default="gbdt",
            choices=["gbdt","dart","rf","goss"],
            help="which boosting type")

    parser.add_argument("--num_leaves",type=int,
            default=31)

    parser.add_argument("--n_estimators",type=int,
            default=100)

    parser.add_argument("--objective",type=str,
            default="multiclass")
    
    parser.add_argument("--subsample",type=float,
            default=0.9,
            help="the ratio of smaple for row,must < 1.0")
    
    parser.add_argument("--subsample_freq",type=int,
            default=100)
    
    parser.add_argument("--colsample_bytree",type=float,
            default=0.9)

    parser.add_argument("--max_depth",type=int,
            default=5,
            help="Max depths for building tree")

    parser.add_argument("--num_iter",type=int,
            default=10000,
            help="number boost rounds to train")

    parser.add_argument("--save_dir",type=str,
            default="./model",
            help="the path for saving model")


    args=parser.parse_args()
    return args

def show_time(diff):
   m, s = divmod(diff, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))


def GridSearch(clf, params, X, y, X_predict, y_predict):
    # Train
    start = time.time()
    model = GridSearchCV(clf, params, scoring='accuracy', n_jobs=5, cv=5).fit(X,y).best_estimator_
    end = time.time()
    print('Training time: ')
    show_time(end - start)

    # Predict
    start = time.time()
    scores=accuracy_score(y_predict, model.predict(X_predict))
    print(scores)

    end = time.time()
    print('Prediction time: ')
    show_time(end - start)
    return model


def Build_Model(args,class_weight,for_cv=False):
    
    boosting_type=args.boosting_type
    num_leaves=args.num_leaves
    max_depth=args.max_depth
    learning_rate=args.learning_rate

    n_estimators=args.n_estimators
    objective=args.objective

    subsample=args.subsample
    subsample_freq=args.subsample_freq if hasattr(args,"subsample_freq") else 0
    colsample_bytree=args.colsample_bytree

    n_jobs=args.n_jobs
    random_state =7777

    model=lgb.LGBMClassifier(boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            class_weight=class_weight,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state)
    if for_cv:
        model=lgb.LGBMClassifier(objective=objective,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight=class_weight)
    return model



def predictor(model_file,data_file):

    with open(model_file,"rb") as fin:
        model=pickle.load(fin)

    with open(data_file,"rb") as fp:
        data=pickle.load(fp)
    fp.close()

    X=data["array"]
    y=data["cluster"]
    features=data["genes"]
    feature_to_key={key:gene for key,gene in enumerate(features)}
    print("There are : {} features in data".format(len(features)))
    print('Feature importances:')
    for key in list(model.feature_importances_)[:20]:
        print("No.{} important feature is {}".format(key,feature_to_key[key]))

    print("Encoder cluster -> label ")
    encoder=LabelEncoder()
    encoder.fit(y)
    label=encoder.transform(y)
    num_classes=len(np.unique(label))

    batch_size=1024
    n_batch=X.shape[0]//batch_size
    i=1
    for count in range(n_batch):
        batch_x=X[i*batch_size:batch_size*(i+1)]
        batch_y=label[i*batch_size:batch_size*(i+1)]
        prob=model.predict(batch_x)
        prediction=np.argmax(prob,1)
        report=classification_report(batch_y,prediction)
        print("No.{} report is : ".format(count+1))
        print(report)
        i+=1

