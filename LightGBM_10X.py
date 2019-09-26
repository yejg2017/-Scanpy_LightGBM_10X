import os
import numpy as np
import argparse
import pickle
import json
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
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

    parser.add_argument("--lr",type=float,
            default=0.1,
            help="the learning ratio for training")


    parser.add_argument("--max_depth",type=int,
            default=5,
            help="Max depths for building tree")

    parser.add_argument("--num_iter",type=int,
            default=10000,
            help="number boost rounds to train")

    parser.add_argument("--save_dir",type=str,
            default="./log",
            help="the path for saving model")


    args=parser.parse_args()
    return args

def show_time(diff):
   m, s = divmod(diff, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))


def train(config):
    with open(config.pkl_file,"rb") as fp:
        data=pickle.load(fp)
    fp.close()

    X=data["array"]
    y=data["cluster"]
    features=data["genes"]
    feature_to_key={key:gene for key,gene in enumerate(features)}
    print("There are : {} features in data".format(len(features)))

    print("Encoder cluster -> label ")
    encoder=LabelEncoder()
    encoder.fit(y)
    label=encoder.transform(y)
    num_classes=len(np.unique(label))
    weight=compute_class_weight("balanced",np.unique(label),label)
    
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=config.test_ratio,shuffle=True)
    print("X train : ({},{})".format(X_train.shape[0],X_train.shape[1]))
    print("X test : ({},{})".format(X_test.shape[0],X_test.shape[1]))
    
    train_data = lgb.Dataset(data=X_train,label=y_train)
    test_data = lgb.Dataset(data=X_test,label=y_test)

    print("Build Model and train")
    params={"objective":"multiclass",
            "num_class":num_classes,
            "num_iterations":config.num_iter,
            "learning_rate":config.lr,
            "num_threads":config.n_jobs,
            "device_type":"cpu",
            "max_depth":config.max_depth,
            "num_leaves":36,
            "min_data_in_leaf":5,
            "lambda_l1":0.05,
            "lambda_l1":0.03,
            "bagging_fraction":0.8,
            "feature_fraction":0.8,
            "bagging_freq": 10,
            "metric_freq":5,
            "metric":["multi_logloss","multi_error"]}
            

    print('Starting training...')
    gbm = lgb.train(params,
                train_data,
                num_boost_round=500,
                valid_sets=test_data,  # eval training data
                #feature_name=features,
                #fobj=loglikelihood,
                learning_rates=lambda iter: config.lr  * (0.99 ** iter))

    print('Feature importances:', list(gbm.feature_importance()))
    print('Saving model...')
    # save model to file
    gbm.save_model(os.path.join(config.save_dir,'model.txt'))
    print('Dumping model to JSON...')
    # dump model to JSON (and save to file)
    model_json = gbm.dump_model()
    
    with open(os.path.join(config.save_dir,'model.json'), 'w+') as f:
        json.dump(model_json, f, indent=4)

    print('Dumping and loading model with pickle...')
    # dump model with pickle
    with open(os.path.join(config.save_dir,'model.pkl'), 'wb') as fout:
        pickle.dump(gbm, fout)

def predictor(model_file,data_file):
    
    with open(model_file,"rb") as fin:
        model=pickle.load(fin)

    with open(data_file,"rb") as fp:
        data=pickle.load(fp)
    fp.close()

    print('Feature importances:', list(model.feature_importance()))

    X=data["array"]
    y=data["cluster"]
    features=data["genes"]
    feature_to_key={key:gene for key,gene in enumerate(features)}
    print("There are : {} features in data".format(len(features)))

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
        

if __name__=="__main__":
    #config=get_config()
    #train(config)
    model_file="log/model.pkl"
    data_dir="log/array.pkl"
    predictor(model_file,data_dir)



