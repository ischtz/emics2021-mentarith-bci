# Analysis functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.stats import linregress

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


def calculate_trial_diff_slopes(samples, skip_samples=0, dur_samples=10000):
    """ Return vector of regression slope for each trial difference trace 
    
    Args:
        skip_samples (int): Starting sample index for slope calculation
        dur_samples (int): Number of samples to include for slope calculation
    """
    half = int(samples.shape[1] / 2)
    t = np.arange(0, dur_samples)
    d = samples[:, half:] - samples[:, 0:half]
    d = d[:, skip_samples:skip_samples+dur_samples] # drop samples from the beginning of trial

    # Slope found via linear regression of remaining samples
    b = np.ones(samples.shape[0]) * np.nan
    for row in np.arange(0, d.shape[0]):
        result = linregress(t, d[row, :])
        b[row] = result.slope
    return b


def classify_trial_slopes(b, thresh=0.0):
    """ Predicts calculation interval 1/2 based on regression slope 
    
    Args:
        thresh (float): Slope threshold (Stoll2013: 0.0)
    """
    r = np.ones(b.shape[0]) * np.nan
    r[b <= thresh] = 1
    r[b > thresh] = 2
    return r


def sdt(responses, ground_truth, labels=None):
    """ Return SDT confusion matrix for a set of responses 
    
    Args:
        responses (array): Predicted response values
        ground_truth (array): True response values
        labels (2-tuple): Actual response labels to use
    """
    if labels is None:
        labels = np.unique(np.array(responses))
    elif len(labels) != 2:
        raise ValueError('Labels must be a two-element vector!')
    
    res = np.array(responses)
    gt = np.array(ground_truth)
    hits = np.sum((gt == labels[0]) & (res == labels[0]))
    miss = np.sum((gt == labels[0]) & (res == labels[1]))
    fa = np.sum((gt == labels[1]) & (res == labels[0]))
    cr = np.sum((gt == labels[1]) & (res == labels[1]))

    return (hits, miss, fa, cr)


def stoll_auc(ds, skip_samples=0, dur_samples=10000):
    """ Return AUC for dataset using Stoll (2013) decoding 
    
    Args:
        skip_samples (int): Starting sample index for slope calculation
        dur_samples (int): Number of samples to include for slope calculation
    """
    tpr, fpr = stoll_roc(ds, skip_samples=skip_samples, dur_samples=dur_samples)
    return(np.trapz(tpr, fpr))


def stoll_roc(ds, skip_samples=0, dur_samples=10000):
    """ Return True- and False Positive Rate nedded to 
    draw ROC curve using the decoding from Stoll et al. (2013) 

    Args:
        skip_samples (int): Starting sample index for slope calculation
        dur_samples (int): Number of samples to include for slope calculation
    """
    b = calculate_trial_diff_slopes(ds.trials, skip_samples=skip_samples, dur_samples=dur_samples)
    slopes = np.sort(b.copy())
    tpr = []
    fpr = []
    for b0 in slopes:
        guess = classify_trial_slopes(b, thresh=b0)
        (hits, miss, fa, cr) = sdt(guess, ds.trials_gt, labels=[1, 2])
        tpr.append(hits / (hits + miss))
        fpr.append(fa / (fa + cr))

    return(np.array(tpr), np.array(fpr))


def stoll_auc_subjects(ds, skip_samples=0, dur_samples=10000):
    """ Calculate AUC for each subject in a dataset 
    using the Stoll (2013) classification method """
    aucs = []
    for sub in np.unique(ds.trials_ppid):
        tpr = []
        fpr = []
        
        tr = ds.trials[ds.trials_ppid == sub, :]
        gt = ds.trials_gt[ds.trials_ppid == sub]
        b = calculate_trial_diff_slopes(tr, skip_samples=skip_samples, dur_samples=dur_samples)
        slopes = np.sort(b.copy())

        for b0 in slopes:
            guess = classify_trial_slopes(b, thresh=b0)
            (hits, miss, fa, cr) = sdt(guess, gt, labels=[1, 2])
            tpr.append(hits / (hits + miss))
            fpr.append(fa / (fa + cr))

        aucs.append(np.trapz(np.array(tpr), np.array(fpr)))
    
    return pd.Series(aucs)


def accuracy(responses, ground_truth):
    """ Accuracy (proportion correct) for a set of response guesses """
    return np.sum(responses == ground_truth) / responses.shape[0]


def stoll_accuracy_subjects(ds, threshold=0.0, skip_samples=0, dur_samples=10000):
    """ Run the analysis from Stoll et al., 2013 on a dataset
    and return accuracy value for each subject as Series """
    b = calculate_trial_diff_slopes(ds.trials, skip_samples=skip_samples, dur_samples=dur_samples)
    guesses = classify_trial_slopes(b, thresh=threshold)
    
    sub_acc = []
    for sub in range(1, 1+len(np.unique(ds.trials_ppid))):
        a = accuracy(guesses[ds.trials_ppid == sub], ds.trials_gt[ds.trials_ppid == sub])
        sub_acc.append(a)
    return pd.Series(sub_acc)


def train_nested_cv_classifiers(samples, gt, outer_folds=7, inner_folds=8):
    """ Train the chosen classifiers using nested cross-validation strategy
    and report accuracy and AUC measures 

    Args:
        samples (ndarray): Pupil sample data
        gt (ndarray): Ground truth class labels
        outer_folds / inner_folds (int): cross-validation folds
    """
    accsT = {}
    aucsT = {}
    paramsT = {}

    # Use for train-test split
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    
    # Use for hyperparameter search
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
    p_grid_knn = {'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]}
    p_grid_svm = {'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                  'C': [1, 10, 100, 1000]}

    # Define classifiers
    lr = LogisticRegressionCV(Cs=10, cv=inner_cv, scoring='roc_auc', refit=True, max_iter=10000)
    knn = KNeighborsClassifier()
    svm = SVC(kernel='rbf', probability=True)
    
    # Outer CV for train/test split
    outer_idx = 1
    for train, test in outer_cv.split(samples, gt):
        #print('CV split {:d}/{:d}'.format(outer_idx, outer_folds))
        x_train = samples[train]
        x_test = samples[test]
        y_train = gt[train]
        y_test = gt[test]

        # We need to evaluate the reference model results on each test subset
        if samples.shape[1] == 20000 or samples.shape[1] == 1200: # 2s at 1 kHz or 60 Hz
            
            # Reference: Stoll (2013), full trace
            Fs = int(samples.shape[1] / 20)
            label = 'Stoll_all'
            if label not in aucsT:
                aucsT[label] = []
            if label not in accsT:
                accsT[label] = []
            if label not in paramsT:
                paramsT[label] = []

            tmpds = namedtuple('Dataset', ['trials', 'trials_gt'])
            tmpds.trials = x_test
            tmpds.trials_gt = y_test
            
            b = calculate_trial_diff_slopes(x_test, skip_samples=0, dur_samples=int(10*Fs))
            guesses = classify_trial_slopes(b, thresh=0.0)
            a = accuracy(guesses, y_test)
            au = stoll_auc(tmpds, skip_samples=0, dur_samples=int(10*Fs))
            accsT[label].append(a)
            aucsT[label].append(au)
            #print('{:s}: acc={:.3f}, auc={:.3f}'.format(label, a, au))

            # Reference: Stoll (2013), 1.5s skip
            label = 'Stoll2013'
            if label not in aucsT:
                aucsT[label] = []
            if label not in accsT:
                accsT[label] = []
            if label not in paramsT:
                paramsT[label] = []

            b = calculate_trial_diff_slopes(x_test, skip_samples=int(1.5*Fs), dur_samples=int(3.5*Fs))
            guesses = classify_trial_slopes(b, thresh=0.0)
            a = accuracy(guesses, y_test)
            au = stoll_auc(tmpds, skip_samples=int(1.5*Fs), dur_samples=int(3.5*Fs))
            accsT[label].append(a)
            aucsT[label].append(au)
            #print('{:s}: acc={:.3f}, auc={:.3f}'.format(label, a, au))
       
        # SVM with rbf kernel
        label = 'SVM'
        if label not in aucsT:
            aucsT[label] = []
        if label not in accsT:
            accsT[label] = []
        if label not in paramsT:
            paramsT[label] = []

        svmCV = GridSearchCV(estimator=svm, param_grid=p_grid_svm, cv=inner_cv, 
                             scoring='roc_auc', refit=True, n_jobs=2)
        svmCV.fit(x_train, y_train)
        a = svmCV.best_estimator_.score(x_test, y_test)
        au = roc_auc_score(y_test, svmCV.best_estimator_.predict_proba(x_test)[:,1])
        accsT[label].append(a)
        aucsT[label].append(au)
        paramsT[label].append(svmCV.best_params_)
        #print('{:s}: acc={:.3f}, auc={:.3f}, params={:s}'.format(label, a, au, str(svmCV.best_params_)))

        # kNN
        label = 'kNN'
        if label not in aucsT:
            aucsT[label] = []
        if label not in accsT:
            accsT[label] = []
        if label not in paramsT:
            paramsT[label] = []

        knnCV = GridSearchCV(estimator=knn, param_grid=p_grid_knn, cv=inner_cv, 
                             scoring='roc_auc', refit=True, n_jobs=2)
        knnCV.fit(x_train, y_train)
        a = knnCV.best_estimator_.score(x_test, y_test)
        au = roc_auc_score(y_test, knnCV.best_estimator_.predict_proba(x_test)[:,1])
        accsT[label].append(a)
        aucsT[label].append(au)
        paramsT[label].append(knnCV.best_params_)
        #print('{:s}: acc={:.3f}, auc={:.3f}, params={:s}'.format(label, a, au, str(knnCV.best_params_)))
        
        # Logistic Regression (built-in CV for Cs)
        label = 'logReg'
        if label not in aucsT:
            aucsT[label] = []
        if label not in accsT:
            accsT[label] = []
        if label not in paramsT:
            paramsT[label] = []
            
        lr.fit(x_train, y_train)
        a = lr.score(x_test, y_test)
        au = roc_auc_score(y_test, lr.decision_function(x_test))
        accsT[label].append(a)
        aucsT[label].append(au)
        paramsT[label].append({'C': lr.C_.tolist()})
        #print('{:s}: acc={:.3f}, auc={:.3f}, params={:s}'.format(label, a, au, str(lr.C_)))

        outer_idx += 1
        
    return (accsT, aucsT, paramsT)


def fit_trial_trace_length_svm(trackers, time_windows=80, outer_folds=7, C=1, gamma=0.0001):
    """ Fit a specified SVM model (C, gamma) for different amounts of
    pupil trace data, using trial traces for each eye tracker. 
    
    Args:
        trackers: List of eye tracker datasets
        time_windows (int): Number of time windows to split trace samples (columns)
        outer_folds (int): Cross-validation folds to use
        C, gamma (float): SVM parameters
    """
    svm_accs = {}
    svm_aucs = {}

    kfcv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    svm = SVC(max_iter=5000, kernel='rbf', C=C, gamma=gamma, probability=True)

    for tracker in trackers:
        svm_accs[tracker.name] = {'m': [], 'max': [], 'min': [], 'sd': []}
        svm_aucs[tracker.name] = {'m': [], 'max': [], 'min': [], 'sd': []}
        
        for window in range(1, time_windows+1):
            win_accs = []
            win_aucs = []
            win_end = window * int(tracker.trials.shape[1] / time_windows)

            # Apply identical cross validation to each time window
            for train, test in kfcv.split(tracker.trials, tracker.trials_gt):
                x_train = tracker.trials[train, 0:win_end]
                x_test = tracker.trials[test, 0:win_end]
                y_train = tracker.trials_gt[train]
                y_test = tracker.trials_gt[test]
                
                svm.fit(x_train, y_train)
                acc = svm.score(x_test, y_test)
                auc = roc_auc_score(y_test, svm.predict_proba(x_test)[:,1])
                win_accs.append(acc)
                win_aucs.append(auc)

            svm_accs[tracker.name]['m'].append(np.mean(win_accs))
            svm_accs[tracker.name]['max'].append(np.max(win_accs))
            svm_accs[tracker.name]['min'].append(np.min(win_accs))
            svm_accs[tracker.name]['sd'].append(np.std(win_accs))
            svm_aucs[tracker.name]['m'].append(np.mean(win_aucs))
            svm_aucs[tracker.name]['max'].append(np.max(win_aucs))
            svm_aucs[tracker.name]['min'].append(np.min(win_aucs))
            svm_aucs[tracker.name]['sd'].append(np.std(win_aucs))
        
        print('{:s} done.'.format(tracker.name))

    return (svm_accs, svm_aucs)


def fit_interval_trace_length_svm(trackers, time_windows=40, outer_folds=7, C=1, gamma=0.0001):
    """ Fit a specified SVM model (C, gamma) for different amounts of
    pupil trace data, using response interval traces for each eye tracker. 
    
    Args:
        trackers: List of eye tracker datasets
        time_windows (int): Number of time windows to split trace samples (columns)
        outer_folds (int): Cross-validation folds to use
        C, gamma (float): SVM parameters
    """
    svm_accs = {}
    svm_aucs = {}

    kfcv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    svm = SVC(max_iter=5000, kernel='rbf', C=C, gamma=gamma, probability=True)

    for tracker in trackers:
        svm_accs[tracker.name] = {'m': [], 'max': [], 'min': [], 'sd': []}
        svm_aucs[tracker.name] = {'m': [], 'max': [], 'min': [], 'sd': []}
        
        for window in range(1, time_windows+1):
            win_accs = []
            win_aucs = []
            win_end = window * int(tracker.periods.shape[1] / time_windows)

            # Apply identical cross validation to each time window
            for train, test in kfcv.split(tracker.periods, tracker.periods_gt):
                x_train = tracker.periods[train, 0:win_end]
                x_test = tracker.periods[test, 0:win_end]
                y_train = tracker.periods_gt[train]
                y_test = tracker.periods_gt[test]
                
                svm.fit(x_train, y_train)
                acc = svm.score(x_test, y_test)
                auc = roc_auc_score(y_test, svm.predict_proba(x_test)[:,1])
                win_accs.append(acc)
                win_aucs.append(auc)

            svm_accs[tracker.name]['m'].append(np.mean(win_accs))
            svm_accs[tracker.name]['max'].append(np.max(win_accs))
            svm_accs[tracker.name]['min'].append(np.min(win_accs))
            svm_accs[tracker.name]['sd'].append(np.std(win_accs))
            svm_aucs[tracker.name]['m'].append(np.mean(win_aucs))
            svm_aucs[tracker.name]['max'].append(np.max(win_aucs))
            svm_aucs[tracker.name]['min'].append(np.min(win_aucs))
            svm_aucs[tracker.name]['sd'].append(np.std(win_aucs))
        
        print('{:s} done.'.format(tracker.name))

    return (svm_accs, svm_aucs)


def fit_final_models(ds, models):
    """ Fit a set of models to a tracker dataset, trials and intervals 
    
    Args:
        ds: Dataset named tuple for one eye tracker 
        models (dict): dict of model objects with labels as keys
    """
    out = {'trials': {}, 'intervals': {}}
    for data in out.keys():
        
        if data == 'trials':
            x_train = ds.trials
            y_train = ds.trials_gt
        else: 
            x_train = ds.periods
            y_train = ds.periods_gt
        Fs = int(x_train.shape[1] / 20)

        # Models from input argument
        for mn in models.keys():
            if mn not in out[data].keys():
                out[data][mn] = {}

            mod = models[mn].fit(x_train, y_train)
            out[data][mn]['model'] = mod
            out[data][mn]['gt'] = y_train
            out[data][mn]['prob'] = mod.predict_proba(x_train)[:,1]
            out[data][mn]['pred'] = mod.predict(x_train)
            out[data][mn]['acc'] = mod.score(x_train, y_train)
            out[data][mn]['auc'] = roc_auc_score(y_train, mod.predict_proba(x_train)[:,1])
            if data == 'trials':
                out[data][mn]['ppid'] = ds.trials_ppid
            else:
                out[data][mn]['ppid'] = ds.periods_ppid
                
            print('{:s}/{:s}, acc={:.3f}, auc={:.3f}'.format(data, mn, out[data][mn]['acc'], out[data][mn]['auc']))
        
        # Stoll et al (2013) model with full / 3.5s data trace
        if data == 'trials':
            for mn in ['Stoll2013', 'Stoll_all']:
                if mn not in out[data].keys():
                    out[data][mn] = {}

                if mn == 'Stoll2013':
                    start = int(1.5*Fs)
                    dur = int(3.5*Fs)
                elif mn == 'Stoll_all':
                    start = 0
                    dur = int(x_train.shape[1] / 2)
                
                out[data][mn]['gt'] = y_train
                out[data][mn]['slope'] = calculate_trial_diff_slopes(x_train, skip_samples=start, dur_samples=dur)
                out[data][mn]['pred'] = classify_trial_slopes(out[data][mn]['slope'], thresh=0.0)
                out[data][mn]['acc'] = accuracy(out[data][mn]['pred'], y_train)
     
                tmpds = namedtuple('Dataset', ['trials', 'trials_gt'])
                tmpds.trials = x_train
                tmpds.trials_gt = y_train
                out[data][mn]['auc'] = stoll_auc(tmpds, skip_samples=start, dur_samples=dur)
                
                if data == 'trials':
                    out[data][mn]['ppid'] = ds.trials_ppid
                else:
                    out[data][mn]['ppid'] = ds.periods_ppid
                
                print('{:s}/{:s}, acc={:.3f}, auc={:.3f}'.format(data, mn, out[data][mn]['acc'], out[data][mn]['auc']))

    return out


def results_to_df(results):
    """ Export a DataFrame of final model results """
    rows = []
    for tracker in ['Eyelink', 'EyeSeeCam', 'EyeTribe']:
        for data in ['trials', 'intervals']:
            for model in ['LogR', 'kNN', 'SVM', 'Stoll2013', 'Stoll_all']:
                if model in results[tracker][data].keys():
                    d = results[tracker][data][model]
                    if data == 'intervals':
                        ppid = np.repeat(d['ppid'], 2)
                    else:
                        ppid = d['ppid']

                    if 'slope' in d.keys():
                        # Only available in Stoll models, others have probability
                        slope = d['slope']
                        prob = np.ones(d['gt'].shape) * np.nan
                    else:
                        slope = np.ones(d['gt'].shape) * np.nan
                        prob = d['prob']

                    for ix, _ in enumerate(d['gt']):
                        r = [tracker, data, model, ppid[ix], d['gt'][ix], d['pred'][ix], prob[ix], slope[ix]]
                        rows.append(r)

    cols = ['device', 'trace', 'model', 'ppid', 'true_val', 'pred_resp', 'pred_prob', 'slope']
    return pd.DataFrame(rows, columns=cols)
