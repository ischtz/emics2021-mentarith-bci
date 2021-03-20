# Load and preprocess raw pupil size data for Mental Arithmetic condition

import os
import sys
import h5py
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.signal import resample
from scipy.interpolate import interp1d
from collections import namedtuple

from pyedfread import edf

data_folder_in =  os.path.join(os.getcwd(), 'data', 'raw')
data_folder_out = os.path.join(os.getcwd(), 'data')

SUBS = list(range(1, 25))
COND = 'MA'

# Preprocessing defaults
BLINK_INTERP = 'linear'


def import_trial_data(sub_csv):
    """ Load a single session file from CSV, select relevant data """

    TRIAL_COLUMNS = {'subject_nr': 'subj',
                     'block_nr': 'block',
                     'count_Question': 'trial',
                     'questionAbout': 'question',
                     'itemIds': 'intervals',
                     'correct_answer': 'resp_truth',
                     'VPanswer': 'resp_online', 
                     'correct_selection': 'correct_online',
                     'currentSlope': 'slope_online',
                     'time_pygaze_start_recording': 't_os_start_trial',
                      'time_calculateMathTask': 't_os_interval1',
                     'time_calculateMathTask_1': 't_os_interval2',
                     'time_select_target': 't_os_select_target'}

    trials = pd.read_csv(sub_csv, sep=',', index_col=False)
    if trials.title[0] != 'mental arithmetic experiment':
        e = '* Error: trial data seems to be from wrong experiment! sub: {:d}'.format(trials.subject_nr[0])
        print(e)
    trials = trials.loc[:, TRIAL_COLUMNS.keys()].rename(TRIAL_COLUMNS, axis='columns')
    trials.loc[:, 'trial'] = trials.loc[:, 'trial'] + 1

    # Calculate ground truth response interval
    trials.loc[:, 'resp_interval'] = 0
    trials.loc[(trials.intervals == 'YN') & (trials.resp_truth == 'Y'), 'resp_interval'] = 1
    trials.loc[(trials.intervals == 'YN') & (trials.resp_truth == 'N'), 'resp_interval'] = 2
    trials.loc[(trials.intervals == 'NY') & (trials.resp_truth == 'Y'), 'resp_interval'] = 2
    trials.loc[(trials.intervals == 'NY') & (trials.resp_truth == 'N'), 'resp_interval'] = 1

    return trials


def import_eyelink_data(sub_edf, eye='left'):
    """ Load a single Eyelink session dataset using pyedfread """
    sam, ev, msg = edf.pread(sub_edf, trial_marker=b'start_trial', filter=['start_trial', 'stop_trial', 'QUESTION'])

    # Select only gaze and pupil data from the correct eye
    if eye == 'left':
        sam.loc[:, 'pa'] = sam.pa_left
        sam.loc[:, 'gx'] = sam.gx_left
        sam.loc[:, 'gy'] = sam.gy_left
    elif eye == 'right':
        sam.loc[:, 'pa'] = sam.pa_right
        sam.loc[:, 'gx'] = sam.gx_right
        sam.loc[:, 'gy'] = sam.gy_right

    # OpenSesame started a new trial right before quitting, drop it here
    msg = msg[~np.isnan(msg.stop_trial_time)]

    return (sam, ev, msg)


def import_eyeseecam_data(tsv_file, eye='left'):
    """ Load a single gaze dataset from EyeSeeCam """

    msg = []
    sam = []
    
    with open(tsv_file, 'r') as tf:
        DF_HEADER = tf.readline().strip().split('\t')
        MSG_HEADER = DF_HEADER[0:2] + ['Message',]
        for row in tf:
            data = row.strip().split('\t')
            
            # Collect messages
            if data[0] == 'LeftSystemTime':
                print('  Double header in file! Using the later dataset.')
                msg = []
                sam = []
                continue
            if len(data) < 5 and data[2] == 'MSG':
                msg.append([float(data[0]), float(data[1]), str(data[3])])
            else:
                sam.append([float(x) for x in data])
    
    msg = pd.DataFrame(msg, columns=MSG_HEADER)
    sam = pd.DataFrame(sam, columns=DF_HEADER)
    
    # Select only gaze and pupil data from the correct eye
    if eye == 'left':
        sam.loc[:, 'pa'] = sam.LeftPupilSize
        sam.loc[:, 'gx'] = sam.LeftEyeHor
        sam.loc[:, 'gy'] = sam.LeftEyeVer
    elif eye == 'right':
        sam.loc[:, 'pa'] = sam.RightPupilSize
        sam.loc[:, 'gx'] = sam.RightEyeHor
        sam.loc[:, 'gy'] = sam.RightEyeVer

    return (sam.copy(), msg.copy())


def import_eyetribe_data(tsv_file, eye='left'):
    """ Load a single gaze dataset from EyeTribe """

    msg = []
    sam = []
    
    with open(tsv_file, 'r') as tf:
        DF_HEADER = tf.readline().strip().split('\t')
        MSG_HEADER = DF_HEADER[0:2] + ['Message',]
        for row in tf:
            data = row.strip().split('\t')

            # Collect messages
            # Eyetribe reports string time stamps
            if len(data) < 5 and data[0] == 'MSG':
                if data[2] == '':
                    # Messages before recording start have no time stamp
                    data[2] = -1
                msg.append([data[1], float(data[2]), str(data[3])])
            else:
                sam.append([data[0], float(data[1]), str(data[2])] + [float(x) for x in data[3:]])

    msg = pd.DataFrame(msg, columns=MSG_HEADER)
    sam = pd.DataFrame(sam, columns=DF_HEADER)

    if eye == 'left':
        sam.loc[:, 'pa'] = sam.Lpsize
        sam.loc[:, 'gx'] = sam.Lrawx
        sam.loc[:, 'gy'] = sam.Lrawy
    elif eye == 'right':
        sam.loc[:, 'pa'] = sam.Rpsize
        sam.loc[:, 'gx'] = sam.Rrawx
        sam.loc[:, 'gy'] = sam.Rrawy

    return (sam.copy(), msg.copy())


def preproc_eyelink_data(sam, ev, msg, blink_gap=100, blink_interp=BLINK_INTERP):
    """ Preprocess Eyelink gaze and pupil data from a single participant dataset """
    
    # Drop excess values (pupil 0.0, gaze -32768.0) due to pyedfread bug
    # https://github.com/ischtz/pyedfread/commit/7f78a2a0eaea219b3d0dd5fc31e82793911dc1ff
    invalid_idx = ((sam.pa == 0.0) & (sam.px_left == -32768.0) & ((sam.px_right == -32768.0)))
    sam.loc[invalid_idx, 'pa'] = np.nan
    sam.loc[invalid_idx, 'gx'] = np.nan
    sam.loc[invalid_idx, 'gy'] = np.nan

    # Drop blinks (set to NaN)
    blinks = ev[ev.blink == True]
    for bl in blinks.iterrows():
        sam.pa[(sam.time >= bl[1].start - blink_gap) & (sam.time <= bl[1].end + blink_gap)] = np.nan
        sam.gx[(sam.time >= bl[1].start - blink_gap) & (sam.time <= bl[1].end + blink_gap)] = np.nan
        sam.gy[(sam.time >= bl[1].start - blink_gap) & (sam.time <= bl[1].end + blink_gap)] = np.nan

    # Interpolate across blink gaps
    sam.pa.interpolate(blink_interp, inplace=True)

    # Normalization
    sam.pa = zscore(sam.pa, nan_policy='omit')

    # Add session time
    sam.loc[:, 't'] = sam.time - sam.time.iloc[0]
    
    return sam, ev, msg


def preproc_eyeseecam_data(sam, msg, blink_gap=20, blink_interp=BLINK_INTERP):
    """ Preprocess EyeSeeCam gaze and pupil data from a single participant dataset """
    
    BLINK_THRESH = 0.00001
    
    # Recalculate trial timing (ESC uses a seconds and a usec field)
    t_start = sam.LeftSystemTime.values[0]
    sam.loc[:, 'Time_ms'] = sam.loc[:, 'LeftSystemTime'] - t_start
    sam.loc[:, 'Time_ms'] = (sam.loc[:, 'Time_ms'] * 1000.0) + np.round(sam.loc[:, 'LeftSystemTime_us'] / 1000.0, 1)

    # Recalculate message timing
    msg.loc[:, 'Time_ms'] = msg.loc[:, 'LeftSystemTime'] - t_start
    msg.loc[:, 'Time_ms'] = (msg.loc[:, 'Time_ms'] * 1000.0) + np.round(msg.loc[:, 'LeftSystemTime_us'] / 1000.0, 1)

    # Drop blinks (set to NaN)    
    # Note that blink gap is specified in #samples, not ms!
    blinks = (sam.pa < BLINK_THRESH).astype(int).diff() # 1: blink start, -1: blink end
    blink_starts = np.where(blinks == 1)[0]
    blink_ends = np.where(blinks == -1)[0]
    sam.pa[sam.pa < BLINK_THRESH] = np.nan
    sam.gx[sam.pa < BLINK_THRESH] = np.nan
    sam.gy[sam.pa < BLINK_THRESH] = np.nan
    for b_start in blink_starts:
        # Apply pre-blink gap
        if b_start >= blink_gap:
            sam.loc[b_start-blink_gap:b_start, 'pa'] = np.nan
            sam.loc[b_start-blink_gap:b_start, 'gx'] = np.nan
            sam.loc[b_start-blink_gap:b_start, 'gy'] = np.nan
        else:
            sam.loc[:b_start, 'pa'] = np.nan
            sam.loc[:b_start, 'gx'] = np.nan
            sam.loc[:b_start, 'gy'] = np.nan
    for b_end in blink_ends:
        # Apply post-blink gap
        if b_end <= sam.pa.shape[0]:
            sam.loc[b_end:b_end+blink_gap, 'pa'] = np.nan
            sam.loc[b_end:b_end+blink_gap, 'gx'] = np.nan
            sam.loc[b_end:b_end+blink_gap, 'gy'] = np.nan
        else:
            sam.loc[b_end:, 'pa'] = np.nan
            sam.loc[b_end:, 'gx'] = np.nan
            sam.loc[b_end:, 'gy'] = np.nan
            
    # Interpolate across blink gaps
    sam.pa.interpolate(blink_interp, inplace=True)

    # Normalization
    sam.pa = zscore(sam.pa, nan_policy='omit')

    # Add session time
    sam.loc[:, 't'] = sam.Time_ms

    return sam, msg


def preproc_eyetribe_data(sam, msg, blink_gap=6, blink_interp=BLINK_INTERP):
    """ Preprocess EyeTribe gaze and pupil data from a single participant dataset """
    BLINK_THRESH = 0.00001
    
    # Drop blinks (set to NaN)    
    # Note that blink gap is specified in #samples, not ms!
    blinks = (sam.pa < BLINK_THRESH).astype(int).diff() # 1: blink start, -1: blink end
    blink_starts = np.where(blinks == 1)[0]
    blink_ends = np.where(blinks == -1)[0]
    sam.loc[sam.pa < BLINK_THRESH, 'pa'] = np.nan
    sam.loc[sam.pa < BLINK_THRESH, 'gx'] = np.nan
    sam.loc[sam.pa < BLINK_THRESH, 'gy'] = np.nan
    for b_start in blink_starts:
        # Apply pre-blink gap
        if b_start >= blink_gap:
            sam.loc[b_start-blink_gap:b_start, 'pa'] = np.nan
            sam.loc[b_start-blink_gap:b_start, 'gx'] = np.nan
            sam.loc[b_start-blink_gap:b_start, 'gy'] = np.nan
        else:
            sam.loc[:b_start, 'pa'] = np.nan
            sam.loc[:b_start, 'gx'] = np.nan
            sam.loc[:b_start, 'gy'] = np.nan
    for b_end in blink_ends:
        # Apply post-blink gap
        if b_end <= sam.pa.shape[0]:
            sam.loc[b_end:b_end+blink_gap, 'pa'] = np.nan
            sam.loc[b_end:b_end+blink_gap, 'gx'] = np.nan
            sam.loc[b_end:b_end+blink_gap, 'gy'] = np.nan
        else:
            sam.loc[b_end:, 'pa'] = np.nan
            sam.loc[b_end:, 'gx'] = np.nan
            sam.loc[b_end:, 'gy'] = np.nan
            
    # Interpolate across blink gaps
    sam.pa.interpolate(blink_interp, inplace=True)
    
    # Normalization
    sam.pa = zscore(sam.pa, nan_policy='omit')
    
    # Add session time
    sam.loc[:, 't'] = sam.time - sam.time[0]

    return sam, msg


def calculate_stimulus_timing(trials, msg, source='eyelink'):
    
    # Calculate onsets for period 1 and 2 in OpenSesame time...
    trials.loc[:, 't_os_onset1'] = trials.t_os_interval1 - trials.t_os_start_trial
    trials.loc[:, 't_os_onset2'] = trials.t_os_interval2 - trials.t_os_start_trial
    
    # ...and in device time
    # Note that QUESTION_time actually represents trial start time in the gaze data file!
    if source == 'eyelink':
        trials.loc[:, 't_tracker_start'] = msg.QUESTION_time.values
    elif source == 'eyeseecam':
        trials.loc[:, 't_tracker_start'] = msg.loc[msg.Message.str.startswith('QUESTION'), :].reset_index(drop=True).Time_ms.values
    elif source == 'eyetribe':
        trials.loc[:, 't_tracker_start'] = msg.loc[msg.Message.str.startswith('QUESTION'), :].reset_index(drop=True).time.values

    trials.loc[:, 't_period1'] = np.round(trials.loc[:, 't_tracker_start'] + trials.loc[:, 't_os_onset1'], 0)
    trials.loc[:, 't_period2'] = np.round(trials.loc[:, 't_tracker_start'] + trials.loc[:, 't_os_onset2'], 0)

    return (trials, msg)


def create_trial_period_array(d, pre_ms=0, post_ms=19999, base_samples=100, 
                              source='eyelink', target_framerate=1000):
    """ Split full trial samples by whether to calculate in the first 
    or second interval (i.e., whether ground truth response is the first 
    or second presented)
    """
    samples = []
    interval = []
    ppid = []
    final_samples = int(((post_ms+pre_ms) / 1000.0) * target_framerate) + 1

    for sub in SUBS:
        s = d[sub]['s']
        for row in d[sub]['t'].iterrows():
            # onset of first calc period in EL time (not question, as there was a keypress start)
            q = row[1].t_period1
            if source == 'eyelink':
                sd = s.loc[(s.time >= q-pre_ms) & (s.time <= q+post_ms), :] # sample data
                if target_framerate == 1000:
                    samples.append(sd.pa.values)

                elif target_framerate != 1000:
                    # Linearly interpolate to target framerate
                    interpolator = interp1d(sd.time.values, sd.pa.values, kind='linear')
                    t_new = np.linspace(sd.time.values.min(), sd.time.values.max(), final_samples)
                    samples.append(interpolator(t_new))

            elif source == 'eyetribe':
                sd = s.loc[(s.time >= q-pre_ms) & (s.time <= q+post_ms), :]

                # Linearly interpolate to target framerate - also for 60 Hz because of too many dropped frames
                interpolator = interp1d(sd.time.values, sd.pa.values, kind='linear')
                t_new = np.linspace(sd.time.values.min(), sd.time.values.max(), final_samples)
                samples.append(interpolator(t_new))

            elif source == 'eyeseecam':
                sd = s.loc[(s.Time_ms >= q-pre_ms) & (s.Time_ms <= q+post_ms), :]
                if target_framerate == 220:
                    samples.append(sd.pa.values)

                elif target_framerate != 220:
                    # Linearly interpolate to target framerate
                    interpolator = interp1d(sd.Time_ms.values, sd.pa.values, kind='linear')
                    t_new = np.linspace(sd.Time_ms.values.min(), sd.Time_ms.values.max(), final_samples)
                    samples.append(interpolator(t_new))

            interval.append(row[1].resp_interval)

        ppid.extend([sub,] * d[sub]['t'].shape[0])
    
    samples = np.array(samples)
    # Baseline-correct trials to first N samples
    for row in range(0, samples.shape[0]):
        samples[row, :] = samples[row, :] - samples[row, 0:base_samples].mean()

    return (samples, np.array(interval), np.array(ppid))


def create_response_period_array(d, post_ms=9999, base_samples=100, source='eyelink',
                                 target_framerate=1000):
    """ Split sample data by "calculate" and "ignore" periods
    independent of the trial they come from """
    samples = []
    respond = []
    ppid = []
    final_samples = int((post_ms / 1000.0) * target_framerate) + 1

    for sub in SUBS:
        s = d[sub]['s']
        for row in d[sub]['t'].iterrows():
            t1 = row[1].t_period1
            t2 = row[1].t_period2
            if source == 'eyelink':
                s1 = s.loc[(s.time >= t1) & (s.time <= t1+post_ms), :] # first period
                s2 = s.loc[(s.time >= t2) & (s.time <= t2+post_ms), :] # second period
                if target_framerate == 1000:
                    samples.append(s1.pa.values)
                    samples.append(s2.pa.values)
                
                elif target_framerate != 1000:
                    intp1 = interp1d(s1.time.values, s1.pa.values, kind='linear')
                    intp2 = interp1d(s2.time.values, s2.pa.values, kind='linear')
                    s1_new = np.linspace(s1.time.values.min(), s1.time.values.max(), final_samples)
                    s2_new = np.linspace(s2.time.values.min(), s2.time.values.max(), final_samples)
                    samples.append(intp1(s1_new))
                    samples.append(intp2(s2_new))

            elif source == 'eyetribe':
                s1 = s.loc[(s.time >= t1) & (s.time <= t1+post_ms), :]
                s2 = s.loc[(s.time >= t2) & (s.time <= t2+post_ms), :]
                intp1 = interp1d(s1.time.values, s1.pa.values, kind='linear')
                intp2 = interp1d(s2.time.values, s2.pa.values, kind='linear')
                s1_new = np.linspace(s1.time.values.min(), s1.time.values.max(), final_samples)
                s2_new = np.linspace(s2.time.values.min(), s2.time.values.max(), final_samples)
                samples.append(intp1(s1_new))
                samples.append(intp2(s2_new))

            elif source == 'eyeseecam':
                s1 = s.loc[(s.Time_ms >= t1) & (s.Time_ms <= t1+post_ms), :]
                s2 = s.loc[(s.Time_ms >= t2) & (s.Time_ms <= t2+post_ms), :]
                if target_framerate == 220:
                    samples.append(s1.pa.values)
                    samples.append(s2.pa.values)

                elif target_framerate != 220:
                    intp1 = interp1d(s1.Time_ms.values, s1.pa.values, kind='linear')
                    intp2 = interp1d(s2.Time_ms.values, s2.pa.values, kind='linear')
                    s1_new = np.linspace(s1.Time_ms.values.min(), s1.Time_ms.values.max(), final_samples)
                    s2_new = np.linspace(s2.Time_ms.values.min(), s2.Time_ms.values.max(), final_samples)
                    samples.append(intp1(s1_new))
                    samples.append(intp2(s2_new))

            if row[1].resp_interval == 1:
                respond.append(1)
                respond.append(0)
            else:
                respond.append(0)
                respond.append(1)

        ppid.extend([sub,] * d[sub]['t'].shape[0])
                
    samples = np.array(samples)
    # Baseline-correct periods individually to avoid dependence on the trial they are from
    for row in range(0, samples.shape[0]):
        samples[row, :] = samples[row, :] - samples[row, 0:base_samples].mean()

    return (samples, np.array(respond), np.array(ppid))


def load_data(data_folder, raw_data_folder='data/raw', file_name='mentarith'):
    """ Load HDF5 data files for analysis, run preprocessing if necessary """
    
    DATA_EL = os.path.join(data_folder, '{:s}_EL.h5'.format(file_name))
    DATA_ES = os.path.join(data_folder, '{:s}_ES.h5'.format(file_name))
    DATA_ET = os.path.join(data_folder, '{:s}_ET.h5'.format(file_name))

    if not os.path.exists(DATA_EL) or not os.path.exists(DATA_ES) or not os.path.exists(DATA_ET):
        print('* At least one HDF5 file is missing, running preprocessing...')
        run_preprocessing(data_folder=data_folder, raw_data_folder=raw_data_folder, file_name=file_name)
    
    # Eyelink
    hfEL = h5py.File(DATA_EL, 'r')
    EL = namedtuple('Dataset', ['trials', 'trials_gt', 'trials_ppid', 'Fs', 'name',
                                'periods', 'periods_gt', 'periods_ppid'])
    EL.Fs = 1000.0
    EL.name = 'Eyelink'
    EL.trials = hfEL['EL']['trials']['samples'][:]
    EL.trials_gt = hfEL['EL']['trials']['interval'][:]
    EL.trials_ppid = hfEL['EL']['trials']['ppid'][:]
    EL.periods = hfEL['EL']['periods']['samples'][:]
    EL.periods_gt = hfEL['EL']['periods']['respond'][:]
    EL.periods_ppid = hfEL['EL']['periods']['ppid'][:]

    # EyeSeeCam
    hfES = h5py.File(DATA_ES, 'r')
    ES = namedtuple('Dataset', ['trials', 'trials_gt', 'trials_ppid', 'Fs', 'name',
                                'periods', 'periods_gt', 'periods_ppid'])
    ES.Fs = 222.0
    ES.name = 'EyeSeeCam'
    ES.trials = hfES['ES']['trials']['samples'][:]
    ES.trials_gt = hfES['ES']['trials']['interval'][:]
    ES.trials_ppid = hfES['ES']['trials']['ppid'][:]
    ES.periods = hfES['ES']['periods']['samples'][:]
    ES.periods_gt = hfES['ES']['periods']['respond'][:]
    ES.periods_ppid = hfES['ES']['periods']['ppid'][:]

    # EyeTribe
    hfET = h5py.File(DATA_ET, 'r')
    ET = namedtuple('Dataset', ['trials', 'trials_gt', 'trials_ppid', 'Fs', 'name',
                                'periods', 'periods_gt', 'periods_ppid'])
    ET.Fs = 60.0
    ET.name = 'EyeTribe'
    ET.trials = hfET['ET']['trials']['samples'][:]
    ET.trials_gt = hfET['ET']['trials']['interval'][:]
    ET.trials_ppid = hfET['ET']['trials']['ppid'][:]
    ET.periods = hfET['ET']['periods']['samples'][:]
    ET.periods_gt = hfET['ET']['periods']['respond'][:]
    ET.periods_ppid = hfET['ET']['periods']['ppid'][:]

    print('* Data loaded.')
    return EL, ES, ET



def run_preprocessing(data_folder, raw_data_folder='data/raw', file_name='mentarith',
                      pre_ms=0, post_ms=19999, base_samples=100, target_framerate=1000):
    """ Run preprocessing for all raw data files and save to HDF5 """
    
    # Temporary pickles to avoid re-importing raw data multiple times
    PICKLE_EL = os.path.join(data_folder, 'rawdata_EL.pkl')
    PICKLE_ES = os.path.join(data_folder, 'rawdata_ES.pkl')
    PICKLE_ET = os.path.join(data_folder, 'rawdata_ET.pkl')
    
    # Pupil trace data for further analysis
    DATA_EL = os.path.join(data_folder, '{:s}_EL.h5'.format(file_name))
    DATA_ES = os.path.join(data_folder, '{:s}_ES.h5'.format(file_name))
    DATA_ET = os.path.join(data_folder, '{:s}_ET.h5'.format(file_name))

    # Eyelink
    if not os.path.exists(DATA_EL):
        if not os.path.exists(PICKLE_EL):
            # Only raw data - import and preprocess
            print('* Eyelink: importing raw data...')
            dEL = {}
            for sub in SUBS:
                sub_edf = os.path.join(raw_data_folder, 'eye_{:s}_EL_{:d}.edf'.format(COND, sub))
                sub_csv = os.path.join(raw_data_folder, 'trial_{:s}_EL_{:d}.csv'.format(COND, sub))
                print(sub_edf)
                
                # Load and preprocess data
                trials = import_trial_data(sub_csv)
                if sub != 20:
                    (samples, events, messages) = import_eyelink_data(sub_edf, eye='left')
                elif sub == 20:
                    # Participant #20 has no left eye data in Eyelink
                    (samples, events, messages) = import_eyelink_data(sub_edf, eye='right')
                (samples, events, messages) = preproc_eyelink_data(samples, events, messages, blink_gap=100)
                (trials, messages) = calculate_stimulus_timing(trials, messages, source='eyelink')

                dEL[sub] = {'s': samples,
                            'e': events,
                            'm': messages,
                            't': trials}

            with open(PICKLE_EL, 'wb') as pf:
                pickle.dump(dEL, pf)
            print('* Eyelink: data imported and pickled.')
            
        elif os.path.exists(PICKLE_EL):
            # Load experiment data from pickle
            with open(PICKLE_EL, 'rb') as pf:
                dEL = pickle.load(pf)
            print('* Eyelink: data found pickled. Delete {:s} to re-process.'.format(PICKLE_EL))

        print('* Eyelink: selecting sample data...')
        (trial_samplesEL, trial_response_intervalEL, trial_ppidEL) = create_trial_period_array(dEL, pre_ms=pre_ms, post_ms=post_ms, source='eyelink', 
                                                                                               base_samples=base_samples, target_framerate=target_framerate)
        (period_samplesEL, period_respondEL, period_ppidEL) = create_response_period_array(dEL, post_ms=9999, base_samples=base_samples, 
                                                                                           source='eyelink', target_framerate=target_framerate)

        with h5py.File(DATA_EL, 'a') as hf:
            dset = hf.create_dataset('EL/trials/samples', trial_samplesEL.shape, dtype='f')
            dset[:] = trial_samplesEL[:]
            dset = hf.create_dataset('EL/trials/interval', trial_response_intervalEL.shape, dtype='f')
            dset[:] = trial_response_intervalEL[:]
            dset = hf.create_dataset('EL/trials/ppid', trial_ppidEL.shape, dtype='f')
            dset[:] = trial_ppidEL[:]

            dset = hf.create_dataset('EL/periods/samples', period_samplesEL.shape, dtype='f')
            dset[:] = period_samplesEL[:]
            dset = hf.create_dataset('EL/periods/respond', period_respondEL.shape, dtype='f')
            dset[:] = period_respondEL[:]
            dset = hf.create_dataset('EL/periods/ppid', period_ppidEL.shape, dtype='f')
            dset[:] = period_ppidEL[:]

        print('* Eyelink: sample data saved to {:s}.'.format(DATA_EL))


    # EyeSeeCam
    if not os.path.exists(DATA_ES):
        if not os.path.exists(PICKLE_ES):
            print('* EyeSeeCam: importing raw data...')
            dES = {}
            for sub in SUBS:
                sub_tsv = os.path.join(raw_data_folder, 'eye_{:s}_ES_{:d}.tsv'.format(COND, sub))
                sub_csv = os.path.join(raw_data_folder, 'trial_{:s}_ES_{:d}.csv'.format(COND, sub))
                print(sub_tsv)
                
                # Load and preprocess data
                trials = import_trial_data(sub_csv)
                (samples, messages) = import_eyeseecam_data(sub_tsv, eye='left')
                (samples, messages) = preproc_eyeseecam_data(samples, messages, blink_gap=20)
                (trials, messages) = calculate_stimulus_timing(trials, messages, source='eyeseecam')

                dES[sub] = {'s': samples,
                            'm': messages,
                            't': trials}

            with open(PICKLE_ES, 'wb') as pf:
                pickle.dump(dES, pf)
            print('* EyeSeeCam: data imported and pickled.')

        elif os.path.exists(PICKLE_ES):
            # Load experiment data from pickle
            with open(PICKLE_ES, 'rb') as pf:
                dES = pickle.load(pf)
            print('* EyeSeeCam: data found pickled. Delete {:s} to re-process.'.format(PICKLE_ES))

        print('* EyeSeeCam: selecting sample data...') # 222 Hz
        (trial_samplesES, trial_response_intervalES, trial_ppidES) = create_trial_period_array(dES, pre_ms=pre_ms, post_ms=post_ms, 
                                                                                               source='eyeseecam', base_samples=base_samples, target_framerate=target_framerate)
        (period_samplesES, period_respondES, period_ppidES) = create_response_period_array(dES, post_ms=9999, source='eyeseecam', 
                                                                                           base_samples=base_samples, target_framerate=target_framerate)

        with h5py.File(DATA_ES, 'a') as hf:
            dset = hf.create_dataset('ES/trials/samples', trial_samplesES.shape, dtype='f')
            dset[:] = trial_samplesES[:]
            dset = hf.create_dataset('ES/trials/interval', trial_response_intervalES.shape, dtype='f')
            dset[:] = trial_response_intervalES[:]
            dset = hf.create_dataset('ES/trials/ppid', trial_ppidES.shape, dtype='f')
            dset[:] = trial_ppidES[:]

            dset = hf.create_dataset('ES/periods/samples', period_samplesES.shape, dtype='f')
            dset[:] = period_samplesES[:]
            dset = hf.create_dataset('ES/periods/respond', period_respondES.shape, dtype='f')
            dset[:] = period_respondES[:]
            dset = hf.create_dataset('ES/periods/ppid', period_ppidES.shape, dtype='f')
            dset[:] = period_ppidES[:]

        print('* EyeSeeCam: sample data saved to {:s}.'.format(DATA_ES))


    # EyeTribe
    if not os.path.exists(DATA_ET):
        if not os.path.exists(PICKLE_ET):
            print('* EyeTribe: importing raw data...')
            dET = {}
            for sub in SUBS:
                sub_tsv = os.path.join(raw_data_folder, 'eye_{:s}_ET_{:d}.tsv'.format(COND, sub))
                sub_csv = os.path.join(raw_data_folder, 'trial_{:s}_ET_{:d}.csv'.format(COND, sub))
                print(sub_tsv)
                
                # Load and preprocess data
                trials = import_trial_data(sub_csv)
                (samples, messages) = import_eyetribe_data(sub_tsv, eye='left')
                (samples, messages) = preproc_eyetribe_data(samples, messages, blink_gap=6)
                (trials, messages) = calculate_stimulus_timing(trials, messages, source='eyetribe')

                dET[sub] = {'s': samples,
                            'm': messages,
                            't': trials}

            with open(PICKLE_ET, 'wb') as pf:
                pickle.dump(dET, pf)
            print('* EyeTribe: data imported and pickled.')

        elif os.path.exists(PICKLE_ET):
            # Load experiment data from pickle
            with open(PICKLE_ET, 'rb') as pf:
                dET = pickle.load(pf)
            print('* EyeTribe: data found pickled. Delete {:s} to re-process.'.format(PICKLE_ET))

        print('* EyeTribe: selecting sample data...') # 60 Hz
        (trial_samplesET, trial_response_intervalET, trial_ppidET) = create_trial_period_array(dET, pre_ms=pre_ms, post_ms=post_ms, 
                                                                                               source='eyetribe', base_samples=base_samples, target_framerate=target_framerate)
        (period_samplesET, period_respondET, period_ppidET) = create_response_period_array(dET, post_ms=9999, source='eyetribe', 
                                                                                           base_samples=base_samples, target_framerate=target_framerate)

        with h5py.File(DATA_ET, 'a') as hf:
            dset = hf.create_dataset('ET/trials/samples', trial_samplesET.shape, dtype='f')
            dset[:] = trial_samplesET[:]
            dset = hf.create_dataset('ET/trials/interval', trial_response_intervalET.shape, dtype='f')
            dset[:] = trial_response_intervalET[:]
            dset = hf.create_dataset('ET/trials/ppid', trial_ppidET.shape, dtype='f')
            dset[:] = trial_ppidET[:]

            dset = hf.create_dataset('ET/periods/samples', period_samplesET.shape, dtype='f')
            dset[:] = period_samplesET[:]
            dset = hf.create_dataset('ET/periods/respond', period_respondET.shape, dtype='f')
            dset[:] = period_respondET[:]
            dset = hf.create_dataset('ET/periods/ppid', period_ppidET.shape, dtype='f')
            dset[:] = period_ppidET[:]

        print('* EyeTribe: sample data saved to {:s}.'.format(DATA_ET))

    print('* Preprocessing done.')
