import os
import numpy as np
import argparse
import pickle
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autoPyTorch import AutoNetClassification
import sklearn.metrics


def get_config():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--pkl_file",type=str,
            default="./log/array.pkl",
            help="the path of pickle file")

    parser.add_argument("--test_ratio",type=float,
            default=0.2,
            help="the ratio of test size")

    parser.add_argument("--batch_size",type=int,
            default=32,
            help="batch size for training")

    parser.add_argument("--n_jobs",type=int,
            default=4,
            help="the number threads for training")

    parser.add_argument("--lr",type=float,
            default=0.05,
            help="the learning ratio for training")

    parser.add_argument("--num_iter",type=int,
            default=5000,
            help="number boost rounds to train")

    parser.add_argument("--save_dir",type=str,
            default="./log/model.pth",
            help="the path for saving model")


    args=parser.parse_args()
    return args


def train(config):
    with open(config.pkl_file,"rb") as fp:
        data=pickle.load(fp)
    fp.close()

    X=data["array"]
    y=data["cluster"]

    print("Encoder cluster -> label ")
    encoder=LabelEncoder()
    encoder.fit(y)
    label=encoder.transform(y)
    num_classes=len(np.unique(label))
    weight=compute_class_weight("balanced",np.unique(label),label)
    
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=config.test_ratio,shuffle=True)
    print("X train : ({},{})".format(X_train.shape[0],X_train.shape[1]))
    print("X test : ({},{})".format(X_test.shape[0],X_test.shape[1]))
    
    print("Build Model")
    model=AutoNetClassification(log_level='info',
            cuda=False,
            dataset_name="VKH_10X",
            shuffle=True,
            num_iterations=config.num_iter,
            budget_type='epochs',
            min_budget=100,
            max_budget=10000,
            result_logger_dir="./logger",
            cross_validator="k_fold", 
            cross_validator_args={"n_splits": 5})
    print("Training")
    model.fit(X_train,y_train,validation_split=0.2)
    return

if __name__=="__main__":
    config=get_config()
    train(config)

    
