# Figures 

import os

import numpy as np

import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9

from analysis import *


# Global color scheme for consistency
C = {'Eyelink':   [Set1_9.mpl_colors[4], Set1_9.mpl_colors[4]],
     'EyeSeeCam': [Set1_9.mpl_colors[1], Set1_9.mpl_colors[1]],
     'EyeTribe':  [Set1_9.mpl_colors[2], Set1_9.mpl_colors[2]]}


def figure1(EL, ES, ET, Fs=1000, auc_results='results/60Hz/stoll_auc.json'):
    """ 
    A: Average pupil trace when calculating in interval 1/2, + experiment timing
    B: Overall and by-subject AUC using the method of Stoll et al. (2013)
    """
    # AUC analysis data (panel B)
    try:
        aucdf = pd.read_json(auc_results)
        AUC = {}
        for row in aucdf.iterrows():
            AUC[row[1]['device']] = [row[1]['ci.min'], row[1]['auc'], row[1]['ci.max']]

    except:
        print('Could not load AUC results, using cached values.')
        AUC = {'Eyelink':   [0.7191, 0.7703, 0.8216],
               'EyeSeeCam': [0.8470, 0.8827, 0.9183],
               'EyeTribe':  [0.7088, 0.7595, 0.8103]}

    fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.8), gridspec_kw={'width_ratios': [3, 1]})
    
    # A: Pupil Trace
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].axvspan(10.0, 20.0, color=(0.8, 0.8, 0.8), alpha=0.4)
    axs[0].set_xlabel('Time relative to first interval onset (s)', fontsize=12)
    axs[0].set_ylabel('Pupil relative to baseline (z)', fontsize=12)
    axs[0].set_xlim([-0.2, 20.2])
    axs[0].set_ylim([-0.8, 0.8])
    
    handles = []
    for ds in [EL, ES, ET]:
        f1 = ds.trials[ds.trials_gt == 1, :]
        f2 = ds.trials[ds.trials_gt == 2, :]
        t = (np.arange(0, f1.shape[1]) / Fs)
    
        m_first = np.mean(f1, axis=0)
        m_second = np.mean(f2, axis=0)
        se_first = np.std(f1, axis=0) / np.sqrt(f1.shape[0])
        se_second = np.std(f2, axis=0) / np.sqrt(f2.shape[0])

        hl, = axs[0].plot(t, m_first, color=C[ds.name][0], linewidth=1.2)
        axs[0].plot(t, m_second, color=C[ds.name][0], linestyle='--', dashes=(3, 2), linewidth=1.2)
        axs[0].fill_between(t, m_first - se_first, m_first + se_first, color=C[ds.name][0], alpha=0.4, linewidth=0)
        axs[0].fill_between(t, m_second - se_second, m_second + se_second, color=C[ds.name][0], alpha=0.4, linewidth=0)

        handles.append(hl)

    axs[0].axvline(0.0, color='k', linestyle=':', linewidth=1.0)
    axs[0].axvline(10.0, color='k', linestyle=':', linewidth=1.0)
    axs[0].legend(handles, [ds.name for ds in [EL, ES, ET]], fontsize=8, loc='lower left')
    axs[0].text(0.5, 0.7, s='First Response Interval', horizontalalignment='left', verticalalignment='center', fontsize=8)
    axs[0].text(10.5, 0.7, s='Second Response Interval', horizontalalignment='left', verticalalignment='center', fontsize=8)


    # B: Stoll (2013) Classification AUC
    axs[1].set_xlim([0.5, 3.5])
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[1].set_xlabel('Eye Tracker', fontsize=12)
    axs[1].set_ylabel('AUC (95% CI)', fontsize=12)
    axs[1].set_ylim([0.0, 1.2])
    axs[1].grid(True, which='major', axis='y')
    axs[1].grid(False, which='major', axis='x')

    v = 1
    for ds in [EL, ES, ET]:

        # By-subject: replication of Stoll (2013) analysis
        auc = stoll_auc_subjects(ds, skip_samples=int(1.5*Fs), dur_samples=int(3.5*Fs))
        axs[1].plot((np.random.random((auc.shape[0], 1)) - 0.5) * 0.35 + v, auc, 'o', 
                 alpha=0.5, markersize=4, color=C[ds.name][0])

        # AUCs derived from using all responses + 95% CI (using pROC in R)
        ci = np.array([AUC[ds.name][1] - AUC[ds.name][0], AUC[ds.name][2] - AUC[ds.name][1]]).reshape(2,1)
        axs[1].plot(v, AUC[ds.name][1], 'ko')
        axs[1].errorbar(v, AUC[ds.name][1], yerr=ci, color='k', linewidth=1.2)
        axs[1].text(v, 1.1, s='{:.2f}'.format(AUC[ds.name][1]), horizontalalignment='center', fontsize=9)

        v += 1

    axs[1].set_xticks(range(1, 4))
    axs[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].set_xticklabels(['EL', 'ESC', 'ET'])
    axs[1].axhline(0.5, color='k', linestyle='--', linewidth=1.0) # chance level
    fig.set_tight_layout(True)


def figure1B_accuracy(EL, ES, ET, Fs=1000):
    """ Overall and by-subject accuracy using the method of Stoll et al. (2013) """
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.8))

    ax.set_xlim([0.5, 3.5])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Eye Tracker', fontsize=12)
    ax.set_ylabel('Accuracy (SE)', fontsize=12)
    ax.set_ylim([0.0, 1.2])
    ax.grid(True, which='major', axis='y')
    ax.grid(False, which='major', axis='x')

    v = 1
    for ds in [EL, ES, ET]:
        
        # By-subject: replication of Stoll (2013) analysis
        acc = stoll_accuracy_subjects(ds, skip_samples=int(1.5*Fs), dur_samples=int(3.5*Fs))
        sem = acc.std() / np.sqrt(acc.shape[0])

        # Individual accuracies
        ax.plot((np.random.random((acc.shape[0], 1)) - 0.5) * 0.35 + v, acc, 'o', 
                 alpha=0.5, markersize=4, color=C[ds.name][0])

        # Average of subject AUCs
        ax.plot(v, acc.mean(), 'ko')
        ax.errorbar(v, acc.mean(), yerr=sem, color='k', linewidth=1.2)
        ax.text(v, 1.1, s='{:.2f}'.format(acc.mean()), horizontalalignment='center', fontsize=9)

        v += 1

    ax.set_xticks(range(1, 4))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['EL', 'ESC', 'ET'])
    
    ax.axhline(0.5, color='k', linestyle='--', linewidth=1.0) # chance level

    fig.set_tight_layout(True)


def figure2(aucs_models, aucs_time, aucs_models_ri, aucs_time_ri):
    """ 
    A: Classification results based on trial traces
    B: Performance of winning classifier (SVM) over time
    """
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.7), gridspec_kw={'width_ratios': [2, 2, 1]})

    # A: Classifier performance over time windows
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].set_xlabel('Trace Length (s)', fontsize=12)
    axs[0].set_ylim([0.4, 1.0])
    axs[0].set_xlim([0, 20.0])
    axs[0].axhline(0.5, color='k', linestyle='--', linewidth=1.0) # chance level
    axs[0].set_ylabel('AUC (+ Range)', fontsize=12)
    axs[0].set_ylim([0.4, 1.0])

    handles = []
    for tn in ['Eyelink', 'EyeSeeCam', 'EyeTribe']:
        t = np.arange(1, len(aucs_time[tn]['m'])+1) * 250.0 / 1000.0
        axs[0].fill_between(t, aucs_time[tn]['min'], aucs_time[tn]['max'], color=C[tn][0], alpha=0.2, linewidth=0)
        hl, = axs[0].plot(t, aucs_time[tn]['m'], '-', color=C[tn][0], markersize=3, linewidth=1.6)
        handles.append(hl)

    axs[0].legend(handles, ['Eyelink', 'EyeSeeCam', 'EyeTribe'], fontsize=8, loc=(0.65, 0.2))
    
    # B: Classifier results
    cls_labels = ['Stoll2013', 'Stoll_all', 'kNN', 'logReg', 'SVM']
    cls2_labels = ['kNN', 'logReg', 'SVM']

    axs[1].set_xlabel('Classification Method', fontsize=12)
    axs[1].set_xlim([0.5, 8.5])
    axs[1].set_ylim([0.4, 1.0])
    axs[1].axhline(0.5, color='k', linestyle='--', linewidth=1.0) # chance level
    axs[1].set_yticklabels([])
    axs[1].grid(True, which='major', axis='y')
    axs[1].grid(False, which='major', axis='x')
    axs[0].text(0.5, 0.95, s='Full Trial', horizontalalignment='left', verticalalignment='center', fontsize=8)
    
    dodge = {'Eyelink': -0.2, 'EyeSeeCam': 0, 'EyeTribe': 0.2}
    for tn in ['Eyelink', 'EyeSeeCam', 'EyeTribe']:
        ix = 1
        for cls in cls_labels:
            crange = np.array([np.max(aucs_models[tn][cls]) - np.mean(aucs_models[tn][cls]),
                              np.mean(aucs_models[tn][cls]) - np.min(aucs_models[tn][cls])]).reshape(2, 1)
            axs[1].plot(ix + dodge[tn], np.mean(aucs_models[tn][cls]), 'o', color=C[tn][0], markersize=4)
            axs[1].errorbar(ix + dodge[tn], np.mean(aucs_models[tn][cls]), yerr=crange, color=C[tn][0], linewidth=1.2)
            ix += 1

        for cls in cls2_labels:
            crange = np.array([np.max(aucs_models_ri[tn][cls]) - np.mean(aucs_models_ri[tn][cls]),
                              np.mean(aucs_models_ri[tn][cls]) - np.min(aucs_models_ri[tn][cls])]).reshape(2, 1)
            axs[1].plot(ix + dodge[tn], np.mean(aucs_models_ri[tn][cls]), 's', color=C[tn][0], markersize=4)
            axs[1].errorbar(ix + dodge[tn], np.mean(aucs_models_ri[tn][cls]), yerr=crange, color=C[tn][0], linewidth=1.2)
            ix += 1

    axs[1].axvline(5.5, linewidth=1.0, color='lightgrey')
    axs[1].set_xticks(range(1, 9))
    axs[1].set_xticklabels(['DS', 'DS/full', 'kNN', 'LogR', 'SVM', 'kNN', 'LogR', 'SVM'])
    
    # C: Classifiers for single response periods
    axs[2].axhline(0.5, color='k', linestyle='--', linewidth=1.0) # chance level
    axs[2].tick_params(axis='both', which='major', labelsize=10)
    axs[2].set_xlabel('Trace Length (s)', fontsize=12)
    axs[2].set_ylim([0.4, 1.0])
    axs[2].set_xlim([0, 10.0])
    axs[2].set_yticklabels([])
    axs[2].text(0.5, 0.95, s='Response Interval', horizontalalignment='left', verticalalignment='center', fontsize=8)
    
    handles = []
    for tn in ['Eyelink', 'EyeSeeCam', 'EyeTribe']:
        t = np.arange(1, len(aucs_time_ri[tn]['m'])+1) * 250.0 / 1000.0
        axs[2].fill_between(t, aucs_time_ri[tn]['min'], aucs_time_ri[tn]['max'], 
                            color=C[tn][0], alpha=0.2, linewidth=0)
        hl, = axs[2].plot(t, aucs_time_ri[tn]['m'], '-', color=C[tn][0], markersize=2, linewidth=1.6)
        handles.append(hl)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03)


def plot_classifier_dicts(dd_dict, measure, ymin=0):
    """ Plot summary of list of dicts (per eye tracker) of classifier measures """
    fig = plt.figure(figsize=(9.8, 3))
    N = len(dd_dict.keys())
    f_idx = 1

    for tracker in dd_dict[measure].keys():
        dd = dd_dict[measure][tracker]
        ax = fig.add_subplot(1, N, f_idx)
        models = list(dd.keys())

        ax.axhline(0.5, color='k', linestyle='--', linewidth=1) # chance level
        for v in range(0, len(models)):
            mrange = np.array([np.mean(dd[models[v]]) - np.min(dd[models[v]]), 
                               np.max(dd[models[v]]) - np.mean(dd[models[v]])]).reshape(2, 1)
            plt.bar(v, np.mean(dd[models[v]]), yerr=mrange, width=0.3)
            ax.text(v, 1.05, s='{:.2f}'.format(np.mean(dd[models[v]])), horizontalalignment='center')

        ax.set_ylim((ymin, 1.2))
        plt.xticks(range(0, len(models)), models)
        ax.set_title(tracker)
        ax.set_xlabel('Model')
        if f_idx == 1:
            ax.set_ylabel('Performance ({:s})'.format(measure))
        f_idx += 1

