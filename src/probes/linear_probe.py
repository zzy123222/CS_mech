from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def fit_binary_probe(X_train, y_train, X_test, y_test, C=1.0, seed=0):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=seed,class_weight="balanced"))
    clf.fit(X_train,y_train); prob=clf.predict_proba(X_test)[:,1]; pred=(prob>=.5).astype(int)
    return {"model":clf,"auc":float(roc_auc_score(y_test,prob)),"accuracy":float(accuracy_score(y_test,pred)),"prob":prob}


def centroid_direction(X_pos, X_neg, normalize=True, eps=1e-8):
    v=np.asarray(X_pos,float).mean(0)-np.asarray(X_neg,float).mean(0)
    if normalize: v=v/(np.linalg.norm(v)+eps)
    return v


def projection_auc(X, y, direction):
    score=np.asarray(X,float)@np.asarray(direction,float)
    return float(roc_auc_score(np.asarray(y),score))
