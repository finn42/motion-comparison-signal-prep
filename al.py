import pandas as pd
import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from scipy import interpolate
from scipy.interpolate import interp1d

import librosa


def local_max_max(cue,cue_delay,sr,thresh = 0.99):
    rms =librosa.feature.rms(y=cue, frame_length=256, hop_length=64, center=True, pad_mode='constant')
    rms_sf = int(sr/64)
    times = librosa.times_like(rms,sr = sr, hop_length=64)
    cue_df = pd.DataFrame(index = times+cue_delay)
    cue_df['rms'] = rms[0]
    cue_df['peaks'] = 0
    a = cue_df['rms'].copy()
    a[a< a.quantile(thresh)] = 0
    peaks =pd.Series(sp.signal.find_peaks(a)[0])#, height = 300, threshold = None, distance=10

    j = 0
    while j <len(peaks):
        seg = peaks.loc[(peaks-peaks[j]).abs()<(rms_sf/4)].copy() # minimum time between peaks set to .25s
        if len(seg)>1:
            j = seg.index[-1]
            peak_j = a.iloc[seg.values].idxmax()
        else:
            peak_j = a.index[peaks[j]]
        cue_df.loc[peak_j,'peaks'] = 1
        j+=1

    return cue_df

def xcov(datax, datay,maxlag=10):
    # https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    rs = []
    for i in range(-maxlag,maxlag):
        rs.append(datax.corr(datay.shift(i)))
    return rs

def min_align(ACC,c_type,cue,prelim_synch_time,max_offs):   
    sampleshift_s = cue['sTime'].diff().median()
    sf = np.round(1/sampleshift_s)
    t_range = [cue['sTime'].iloc[0],cue['sTime'].iloc[-1]]
    xrange = [pd.to_timedelta(t_range[0],unit = 's') + prelim_synch_time,
              pd.to_timedelta(t_range[1],unit = 's') + prelim_synch_time]
#     print(xrange)
    sig_sTime = cue['sTime'].values #np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)
    cue.loc[:,'dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time

    max_offs = 5
    X = dt_cut(ACC,'dev_dTime',xrange[0],xrange[1])
    sig_t = (X['dev_dTime'].dt.tz_localize(None) - prelim_synch_time.tz_localize(None)).dt.total_seconds()
    sig_v = X['Jerk']
    f = interpolate.interp1d(sig_t,sig_v,fill_value='extrapolate')
    new_sig = f(sig_sTime)
    signal = pd.DataFrame()
    signal.loc[:,'Jerk'] = new_sig
    signal.loc[signal['Jerk'].isna(),'Jerk'] = 0
    # scale signals a little 
    M = signal['Jerk'].quantile(0.998)
    signal.loc[:,'Jerk']  = signal['Jerk']/M
    signal.loc[signal['Jerk']>1,'Jerk'] = 1
    signal.loc[signal['Jerk']<0,'Jerk'] = 0
    signal.loc[:,'sTime'] = sig_sTime
    signal.loc[:,'dev_dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time.tz_localize(None)
    length = np.min([len(signal),len(cue)]) # they should match, but just in case

    xcorred = pd.DataFrame()
    max_offs = 5
    xcorred['shift_s'] = np.linspace(-max_offs,max_offs,int(2*max_offs*sf), endpoint=False) 
    xcorred['r'] = xcov(cue[c_type].iloc[:length], signal['Jerk'].iloc[:length],int(max_offs*sf))
    max_shift = xcorred.loc[xcorred['r'].argmax(),'shift_s']
    max_r = xcorred['r'].max()
    cue.loc[:,'dev_dTime'] = cue['dTime'] - pd.to_timedelta(max_shift,unit='s')

    fig, axes = plt.subplots(3,1,figsize=(12,5))
    ax =axes[0]
    cue.plot(x='sTime',y=c_type,ax=ax)
    signal.plot(x='sTime',y='Jerk',label='ACC',ax=ax)
    ax.set_title('ACC synch alignment')
    ax.set_ylabel('Unaligned')
    ax.legend()

    ax = axes[1]
    xcorred.plot(x='shift_s',y='r',ax=ax)
    ax.plot(max_shift,max_r,'ro')
    ax.set_ylim([0,1])
    ax.grid(True)
    ax.set_xticklabels('')

    ax = axes[2]
    ax.plot(cue['dev_dTime'],cue[c_type],label=c_type)
    ax.plot(signal['dev_dTime'],signal['Jerk'],label='ACC')
    ax.xaxis.set_tick_params(rotation=40)

    ax.grid(True)
    ax.set_title('shift '+ str(np.round(max_shift,3))+ ' s')
    ax.set_ylabel('Aligned')
    ax.set_xlabel('dev_Time')
    plt.show()

    cue_time = cue.loc[cue['peaks']==1,'dev_dTime'].iloc[0].tz_localize(None) 
    C_results = {'best_dt': cue_time,'best_s': max_shift,'best_r': max_r,'CCC':xcorred,'cue':cue,'Jerk':signal}
    # C_results = {'best_dt': cue_time,'best_s': max_shift,'best_r': max_r,'CCC':xcorred,'cue':cue,'signal':signal}

    return C_results

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def cue_template_make(peak_times,sf, t_range):
    # peak times is list of time points for onsets to clapping or tapping sequence, in seconds
    # sf is sample frequency in hz
    # buffer is duration of zeros before and after the peak times in generated template, in seconds
    
    peaks = np.array(peak_times)
    c_start = t_range[0]+peaks[0]
    c_end = t_range[1]+peaks[0]
    
    cue_sTime = np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)+peaks[0]

    cue = pd.DataFrame()
    cue['sTime'] = cue_sTime
    cue['peaks'] = 0
    cue['taps'] = 0
    cue['claps'] = 0
    cue['slowtable'] = 0
    
    for pk in peak_times:
        cue.loc[find_nearest_idx(cue['sTime'],pk),'peaks'] = 1
        
    roll_par = int(0.2*sf)
    sum_par = int(0.05*sf)
    ewm_par = int(0.1*sf)
    cue['slowtable'] =2*cue['peaks'].ewm(span = ewm_par).mean()+ 0.6*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)        

    roll_par = int(0.05*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.04*sf)
    cue['claps'] =2*cue['peaks'].ewm(span = ewm_par).mean()+ 0.6*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    roll_par = int(0.02*sf)
    sum_par = int(0.02*sf)
    ewm_par = int(0.02*sf)
    cue['taps'] =1.5*cue['peaks'].ewm(span = ewm_par).mean()+ 0.5*cue['peaks'].rolling(roll_par, win_type='gaussian',center=True).sum(std=2)
    
    cue[cue.isna()] = 0
    return cue


def dt_cut(V,dt_col,t1,t2):
    V[dt_col] = pd.to_datetime(V[dt_col])
    X = V.loc[V[dt_col]>t1,:].copy()
    X = X.loc[X[dt_col]<t2,:].copy()
    if len(X)<1:
        print('Recording does not intersect with that time interval')
        return
    else:
        return X

def min_align_noplot(ACC,cue,prelim_synch_time,max_offs):
    # sf infered from cue
    sampleshift_s = cue['sTime'].diff().median()
    sf = np.round(1/sampleshift_s) 
    t_range = [cue['sTime'].iloc[0],cue['sTime'].iloc[-1]]
    c_type = cue.columns[1]
    
    xrange = [pd.to_timedelta(t_range[0],unit = 's') + prelim_synch_time,pd.to_timedelta(t_range[1],unit = 's') + prelim_synch_time]
    sig_sTime = cue['sTime'].values #np.linspace(t_range[0],t_range[1],sf*(t_range[1]-t_range[0]),endpoint=False)
    
    # add preliminary time stamples to cue
    cue.loc[:,'dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time
    
    # ACC signal excerpt at correct sample rate
    X = ACC.loc[ACC['dev_dTime']<xrange[1],:].copy()
    X = X.loc[X['dev_dTime']>=xrange[0],:].copy()
    sig_t = (X['dev_dTime'].dt.tz_localize(None) - prelim_synch_time.tz_localize(None)).dt.total_seconds()
    sig_v = X['Jerk']
    f = interpolate.interp1d(sig_t,sig_v,fill_value='extrapolate')
    new_sig = f(sig_sTime)
    signal = pd.DataFrame()
    signal.loc[:,'Jerk'] = new_sig
    signal.loc[signal['Jerk'].isna(),'Jerk'] = 0
    # scale signals a little 
    M = signal['Jerk'].quantile(0.998)
    signal.loc[:,'Jerk']  = signal['Jerk']/M
    signal.loc[signal['Jerk']>1,'Jerk'] = 1
    signal.loc[signal['Jerk']<0,'Jerk'] = 0
    signal.loc[:,'sTime'] = sig_sTime
    signal.loc[:,'dev_dTime'] = pd.to_timedelta(sig_sTime,unit='s')+prelim_synch_time

    length = np.min([len(signal),len(cue)]) # they should match, but just in case
#     CCC = ax2.xcorr(cue[c_type].iloc[:length], signal['Jerk'].iloc[:length], usevlines=True, , normed=True, lw=3)
    x = sp.signal.detrend(np.asarray(cue[c_type].iloc[:length]))
    y = sp.signal.detrend(np.asarray(signal['Jerk'].iloc[:length]))
    maxlags=int(max_offs*sf)
    c = np.correlate(x, y, mode=2)
    CCC = [[],[]]
    CCC[0]= np.arange(-maxlags, maxlags + 1)
    CCC[1] = c[length - 1 - maxlags:length + maxlags]

    cue.loc[:,'dev_dTime'] = cue['dTime'] - pd.to_timedelta(sampleshift_s*CCC[0][np.argmax(CCC[1])],unit='s')

    cue_time = cue.loc[find_nearest_idx(cue['sTime'], 0),'dev_dTime']
    C_results = {'best': cue_time,'CCC':CCC,'cue':cue,'Jerk':signal}
    return C_results


def test_shift(Res,shifting):
    alt_cue = Res['cue'].copy()
    if pd.isnull(shifting):
        print('is nan')
        return
    else:
        if isinstance(shifting,float):
            dts = shifting
        else:
            if isinstance(shifting, dt.datetime):
                cue_zero = alt_cue.loc[find_nearest_idx(alt_cue['sTime'], 0),'dTime']
                dts = (cue_zero-shifting).total_seconds()
        c_type = alt_cue.columns[1]
        fig = plt.figure(figsize=(15,3))
        ax1 = plt.subplot(211)
        alt_cue.plot.line(x='sTime',y=c_type,ax=ax1)
        Res['Jerk'].plot(x='sTime',y='Jerk',label='ACC',ax=ax1,)
        ax1.set_ylabel('Unaligned')
        ax1.legend()
        alt_cue.loc[:,'dev_dTime'] =  Res['cue']['dTime'] - pd.to_timedelta(dts,unit='s')

        cue_time = alt_cue.loc[find_nearest_idx(alt_cue['sTime'], 0),'dev_dTime']
        dt_sh = pd.to_timedelta(7,unit='s')

        ax1 = plt.subplot(212)
        alt_cue.plot.line(x='dev_dTime',y=c_type,ax=ax1)
        Res['Jerk'].plot(x='dev_dTime',y='Jerk',label='ACC',ax=ax1,)
        ax1.set_xlim([cue_time-dt_sh/2,cue_time+dt_sh*1.5])
        ax1.grid(True)
        ax1.set_title('shift '+ str(dts)+ ' s')
        plt.show()

        return cue_time

def alt_xc_peaks(Res,ccthresh):
    CCC = Res['CCC']
    cue = Res['cue']
    mid_off = int((len(CCC[0])-1)/2)
    sf = np.round(1/cue['sTime'].diff().median())

    V =np.clip(CCC[1], ccthresh, 1)
    pks = pd.DataFrame()
    pks['ind'] = argrelextrema(V, np.greater)[0]
    pks['corr']= V[argrelextrema(V, np.greater)]
    pks['shift s'] = (pks['ind']-mid_off)/sf
    return pks

def xcov(datax, datay,maxlag=10):
    # https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    rs = []
    for i in range(-maxlag,maxlag):
        rs.append(datax.corr(datay.shift(i)))
    return rs