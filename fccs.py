from plmodels import *
import pandas as pd
import numpy as np

def create_IP_df(y_test, clf_prediction, anom_prediction, clf_proba, anom_proba):
    out_df = pd.DataFrame()
    out_df['true_label'] = y_test
    out_df['predicted_label'] = clf_prediction
    out_df['anom_label'] = anom_prediction
    out_df['anom_probabilities'] = [list(row) for row in anom_proba]
    out_df['Predicted_FCC'] = out_df.apply(lambda row: row['predicted_label'] if row['anom_label'] == 1 else 'omission',axis=1)
    out_df['probabilities'] = [list(row) for row in clf_proba]
    out_df['Max probability'] = np.max(clf_proba,axis = 1)
    out_df = out_df.drop(columns=['anom_label'])
    return out_df

def create_OP_df(y_test, clf_prediction, clf_proba, threshold):
    out_df = pd.DataFrame()
    out_df['true_label'] = y_test
    out_df['predicted_label'] = clf_prediction
    out_df['probabilities'] = [list(row) for row in clf_proba]
    out_df['Max probability'] = np.max(clf_proba,axis = 1)
    out_df['Predicted_FCC'] = out_df.apply(
        lambda row: row['predicted_label'] if row['Max probability'] > threshold else 'omission',
        axis=1
    )
    return out_df

def create_SW_df(y_test, anom_prediction, clf_prediction, clf_proba, threshold, anom_proba):
    out_df = pd.DataFrame()
    out_df['true_label'] = y_test
    out_df['predicted_label'] = clf_prediction
    out_df['anom_label'] = anom_prediction
    out_df['anom_probabilities'] = [list(row) for row in anom_proba]
    # out_df['probabilities'] = pd.DataFrame({'probabilities': [list(row) for row in clf_proba]})
    out_df['probabilities'] = [list(row) for row in clf_proba]

    out_df['Max probability'] = np.max(clf_proba,axis = 1)
    out_df['Predicted_FCC'] = out_df.apply(
        lambda row: row['predicted_label'] if row['anom_label'] == 1 and row['Max probability'] > threshold else 'omission',
        axis=1
    )
    out_df = out_df.drop(columns=['anom_label'])
    print("Fahad")

    return out_df

def IP(clf, anomaly_clf, dataset, y_test):

    anom_prediction = anomaly_clf.predict(dataset)
    clf_prediction = clf.predict(dataset)
    anom_proba = anomaly_clf.predict_proba(dataset)
    clf_proba = clf.predict_proba(dataset)
    df = create_IP_df(y_test=y_test, clf_prediction=clf_prediction, anom_prediction = anom_prediction, clf_proba= clf_proba, anom_proba = anom_proba)

    return df

def OP(clf, dataset, y_test, threshold = 0.90):
    clf_prediction = clf.predict(dataset)
    clf_proba = clf.predict_proba(dataset)
    df = create_OP_df(y_test=y_test, clf_prediction=clf_prediction, clf_proba= clf_proba, threshold=threshold)
    return df

def SW(clf, anomaly_clf, dataset, y_test, threshold = 0.90):
    anom_prediction = anomaly_clf.predict(dataset)
    anom_proba = anomaly_clf.predict_proba(dataset)
    clf_prediction = clf.predict(dataset)
    clf_proba = clf.predict_proba(dataset)
    df = create_SW_df(y_test=y_test, anom_prediction= anom_prediction, clf_prediction=clf_prediction, clf_proba= clf_proba, threshold=threshold, anom_proba = anom_proba)
    return df