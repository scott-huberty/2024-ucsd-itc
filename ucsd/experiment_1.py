import os
from glob import glob
from collections import OrderedDict
from pathlib import Path
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import mne
from mne.preprocessing import read_ica_eeglab
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
from mne.preprocessing import read_ica_eeglab
# from mne.time_frequency import psd_welch

# from acareeg.visualization import plot_values_topomap




eog_channels = ["E1", "E8", "E14", "NAS", "E21", "E25", "E32", "E125", "E126", "E127", "E128"]
chan_mapping = {"E17": "NAS", "E22": "Fp1", "E9": "Fp2", "E11": "Fz", "E124": "F4", "E122": "F8", "E24": "F3",
                "E33": "F7", "E36": "C3", "E45": "T7", "E104": "C4", "E108": "T8", "E52": "P3", "E57": "LM",
                "E58": "P7", "E92": "P4", "E100": "RM", "E96": "P8", "E62": "Pz", "E70": "O1", "E75": "Oz",
                "E83": "O2"}

#Making a dictionary object out of a standard Montage
ch_positionss = mne.channels.make_standard_montage('GSN-HydroCel-129').get_positions()
#renaming dictionary key Cz to E129 to match my channel names
old_key = "Cz"
new_key = "E129"
ch_positionss['ch_pos'][new_key] = ch_positionss['ch_pos'].pop(old_key)
#making a new montage from my edited dictionary
ucsd_montage = mne.channels.make_dig_montage(ch_pos=ch_positionss['ch_pos'],
                              nasion=ch_positionss['nasion'],
                              lpa=ch_positionss['lpa'],
                              rpa=ch_positionss['rpa'],
                              hsp=None, hpi=None,
                              coord_frame='unknown')


#(temporary) event mapping in processed .set files
custom_mapping = {f'mov+_{i}' : i for i in (1,2,3,4,5,6,7)}
custom_mapping['visual_contact'] = 8

#New readable Event dictionary for epoching

readable_event_dict_hed = {'experiment1':{'mov+/audio_only': 1,
                                          'mov+/sync_av/left': 2,
                                          'mov+/async_av/left': 3,
                                          'mov+/video_only/left': 4,
                                          'mov+/sync_av/right': 5,
                                          'mov+/async_av/right': 6,
                                          'mov+/video_only/right': 7,
                                          'visual_contact' : 8},
                           'experiment2': {'mov+/audio_only': 1,
                                           'mov+/sync_av/left': 2,
                                           'mov+/async_av/left': 3,
                                           'mov+/occlusion_av/left': 4,
                                           'mov+/sync_av/right': 5,
                                           'mov+/async_av/right': 6,
                                           'mov+/occlusion_av/right': 7}
                          }
                 

def preprocess(raw, notch_width=None, line_freq=60.0):
    if notch_width is None:
        notch_width = np.array([1.0, 0.1, 0.01, 0.1])

    notch_freqs = np.arange(line_freq, raw.info["sfreq"]/2.0, line_freq)
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], fir_design='firwin',
                     notch_widths=notch_width[:len(notch_freqs)], verbose=None)


def mark_bad_channels(raw, file_name, mark_to_remove=("manual", "rank")):
    raw_eeg = _check_load_mat(file_name, None)
    info, _, _ = _get_info(raw_eeg)
    chan_info = raw_eeg.marks["chan_info"]

    mat_chans = np.array(info["ch_names"])
    assert (len(chan_info["flags"][0]) == len(mat_chans))

    if len(np.array(chan_info["flags"]).shape) > 1:
        ind_chan_to_drop = np.unique(np.concatenate([np.where(flags)[0]
                                                     for flags, label
                                                     in zip(chan_info["flags"],
                                                            chan_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_chan_to_drop = np.where(chan_info["flags"])[0]

    bad_chan = [chan for chan in mat_chans[ind_chan_to_drop]]
    print(f'flagged chans: {bad_chan}')

                     
                     
def add_bad_segment_annot(raw, file_name, mark_to_remove=("manual",)):
    raw_eeg = _check_load_mat(file_name, None)
    time_info = raw_eeg.marks["time_info"]

    if len(np.array(time_info["flags"]).shape) > 1:
        ind_time_to_drop = np.unique(np.concatenate([np.where(flags)[0] 
                                                    for flags, label
                                                    in zip(time_info["flags"],
                                                           time_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_time_to_drop = np.where(time_info["flags"])[0]

    ind_starts = np.concatenate(
        [[ind_time_to_drop[0]], ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0] + 1]])
    ind_ends = np.concatenate([ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0]], [ind_time_to_drop[-1]]])
    durations = (ind_ends + 1 - ind_starts) / raw.info["sfreq"]
    onsets = ind_starts / raw.info["sfreq"]

    for onset, duration in zip(onsets, durations):
        raw.annotations.append(onset, duration, description="bad_lossless_qc")
        print('flagged time: ', onset, 'duration', duration)
                                         
                     
                     
def remove_rejected_ica_components(raw, file_name, inplace=True):
    raw_eeg = _check_load_mat(file_name, None)
    mark_to_remove = ["manual"]
    comp_info = raw_eeg.marks["comp_info"]

    if len(np.array(comp_info["flags"]).shape) > 1:
        ind_comp_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(comp_info["flags"],
                                                                                                comp_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_comp_to_drop = np.where(comp_info["flags"])[0]
        
    print(f'ind_comp_to_drop: {ind_comp_to_drop}')

    if inplace:
        read_ica_eeglab(file_name).apply(raw, exclude=ind_comp_to_drop)
    else:
        read_ica_eeglab(file_name).apply(raw.copy(), exclude=ind_comp_to_drop)

                                
                     
def preprocessed_raw(path, line_freq, montage=ucsd_montage, verbose=False, rename_channel=False, apply_ica=True,
                     interp_bad_ch=True, reset_bads=True):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=verbose)
    
    raw.set_montage(montage, verbose=verbose)

    preprocess(raw, line_freq=line_freq, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))

    raw = raw.filter(1, None, fir_design='firwin', verbose=verbose)

    mark_bad_channels(raw, path)
    add_bad_segment_annot(raw, path)
    if apply_ica:
        remove_rejected_ica_components(raw, path, inplace=True)

    if interp_bad_ch:
        raw = raw.interpolate_bads(reset_bads=reset_bads, verbose=verbose)

    if rename_channel:
        raw.rename_channels({ch: ch2 for ch, ch2 in chan_mapping.items() if ch in raw.ch_names})

    raw.set_channel_types({ch: "eog" for ch in eog_channels if ch in raw.ch_names})

    return raw
                       

def get_contact_annotations(raw):
    visual_contacts = []

    for event in raw.annotations : 
        if event['description'] in {'mov+_1','mov+_2', 'mov+_3', 'mov+_4', 'mov+_5', 'mov+_6','mov+_7'} : 
            if event['description'] in {'mov+_3', 'mov+_6'} :
                visual_contacts.append(event['onset'] + .775)

            else :
                visual_contacts.append(event['onset'] + 1.225)
    
    visual_contact_onsets = [event['onset'] + .775 if event['description'] in ['mov+_3', 'mov+_6']
                         else event['onset'] + 1.225
                         for event in raw.annotations
                         if event['description'] in ['mov+_1','mov+_2','mov+_3','mov+_4', 'mov+_5','mov+_6','mov+_7']]
    
    visual_contact_durations = [0.5 for each_event in visual_contact_onsets]
    visual_contact_descriptions = ['visual_contact' for each_event in visual_contact_onsets]
    visual_contact_annotations = mne.Annotations(onset = visual_contact_onsets,
                                                 duration = visual_contact_durations,
                                                 description = visual_contact_descriptions)
    
    raw2 = raw.copy().set_annotations(raw.annotations + visual_contact_annotations)
    
    return raw2

def add_system_offset(raw):
    #adding 35ms to the onset of each event to account for system offset
    #--------------------------------------------------------------------
    for i, ann in enumerate(raw.annotations) :
        raw.annotations.onset[i] += .035
        
def get_epochs(raw, experiment, tmin=-.2, tmax=2.00, baseline=(None,0)):
    (events_from_annot, event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
    epochs = mne.Epochs(raw,
                        events=events_from_annot,
                        event_id=readable_event_dict_hed[experiment],
                        tmin=tmin, tmax=tmax, baseline=baseline,
                        preload=True)
    return epochs
    

def get_psd_array(epochs):
    #psd.mean average across dimension 0 i.e. epochs; .T transposes the sheet so chan names is columns
    #pick selects only the EEG channel names, not eog channels

    srate = epochs.info['sfreq'] #500 samples per second

    nfft = int(2 * srate) 
    '''#500 samples per second, for a total of 1000 samples, or 2 seconds; the same size as each epoch.
    1 FFT per epoch'''

    welch_length = int(0.5*nfft)
    '''1 welch segment per 500 samples, making 2 welch segments per fft/epoch. 
    This captures my lowest frequency of interest, 2hz, twice per welch segment, which are 1 second long. 
    using more welch windows results in a smoother psd (noise is averaged out). 
    So I am using as many sliding welch segments as possible while still being long enough to encompass 
    two full cycles of my lowest freq of interst (2hz), following convention''' 

    overlap = (0.25*nfft) #250 points overlap, which makes a 50% of each welch segment (500 points)


    '''Using the above variables as input to my function. So for each 2-second epoch, 
    I am calculating a 2-second FFT, with a 1-second sliding welch segment, 
    with 50% overlap between welch segments.'''
    psd, freqs = psd_welch(epochs, fmin=2, n_fft=nfft, n_per_seg=welch_length, n_overlap = (0.25*nfft), average='mean')


    #first argument is the data,  
    array = xr.DataArray(psd,
         dims = ['epochs','channels','freqs'],
         coords = {'epochs': np.arange(psd.shape[0]),
                  'channels': epochs.copy().pick('eeg').ch_names,
                  'freqs': freqs})

    return array


def xr_preprocess(ds, variable, remove_skirt_chans=True):
    '''keep only the selected variable and coords for each file'''
    if remove_skirt_chans:
        eog = ['E1', 'E8', 'E14', 'E17', 'E21', 'E25', 'E32', 'E43','E48', 'E49', 'E56', 'E63','E68',
               'E73','E81', 'E88','E94','E107', 'E113','E119', 'E120', 'E125', 'E126', 'E127', 'E128']
    return ds[variable].sel(frequency=slice(2,20),
                                           channel=[chan for chan
                                           in ds.channel.values
                                           if chan not in eog],
                                           ).squeeze()

def read_netcdfs(files, dim, variable, transform_func=xr_preprocess):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds, variable=variable)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined



def add_trial_information(y_text, condition, fontsize=8, color='k', ax=None):
    if condition == 'sync_av':
        x_times = [1.45]
        x_conditions = ['CONTACT\n + SOUND']
        if ax is None: ax = plt.gca()
        plt.sca(ax)
        plt.axvline(1.25, lw=2, color='r')
        for x_t, t_t in zip(x_times, x_conditions):
            plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                     va='center', fontweight='bold')
    if condition == 'async_av':
        x_times = [0.95, 1.45]
        x_conditions = ['BOUNCE\nSOUND', 'VISUAL\nCONTACT']
        if ax is None: ax = plt.gca() 
        plt.sca(ax)
        plt.axvline(0.75, lw=2, color='r')
        plt.axvline(1.25, lw=2, color='r')
        for x_t, t_t in zip(x_times, x_conditions):
            plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                     va='center', fontweight='bold')
            
    if condition == 'audio_only':
        x_times = [1.45]
        x_conditions = ['BOUNCE\nSOUND']
        if ax is None: ax = plt.gca() 
        plt.sca(ax)
        plt.axvline(1.25, lw=2, color='r')
        for x_t, t_t in zip(x_times, x_conditions):
            plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                     va='center', fontweight='bold')
            
    if condition == 'video_only':
            x_times = [1.45]
            x_conditions = ['VISUAL\nCONTACT']
            if ax is None: ax = plt.gca() 
            plt.sca(ax)
            plt.axvline(1.25, lw=2, color='r')
            for x_t, t_t in zip(x_times, x_conditions):
                plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                         va='center', fontweight='bold')
                
    if condition == 'occlusion_av':
            x_times = [0.80, 1.45]
            x_conditions = ['OCCLUSION','VISUAL\nCONTACT']
            if ax is None: ax = plt.gca() 
            plt.sca(ax)
            plt.axvline(0.6, lw=2, color='r')
            plt.axvline(1.25, lw=2, color='r')
            for x_t, t_t in zip(x_times, x_conditions):
                plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                         va='center', fontweight='bold')
                
                
def itc_plot_by_condition(dataset, experiment, frequencies, times, roi, y_text=18, vmin=None, vmax=None):
    ''''returns: 
            a plot of the alpha-gamma pac for each condition
      parameters:
          dataset (xarray data_array): The PAC array that you want to plot.
          roi (list): channels that you want to include
          vmax (int): maximum value to anchor the heatmap. adjust as needed.
          space (str): 'channel' or 'source'
          '''
        
     # define the figure canvas..
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8),
                             sharex=True, sharey=True)
    # Loop over conditions..
    conditions = {'experiment1': enumerate(['video_only','audio_only','sync_av','async_av']),
                  'experiment2': enumerate(['occlusion_av','audio_only','sync_av','async_av'])}
    
    for index, condition in tqdm(conditions[experiment], total=4, desc='plotting... '): 
        # load, prep, and plot data..
        (dataset.sel(channel=roi,
                     condition=condition,
                     time=times,
                     frequency=frequencies)
                .mean(['ID','channel'])
                .squeeze()
                .plot(ax=axes.flatten()[index],vmin=vmin, vmax=vmax))
        add_trial_information(y_text, condition=condition, color='white', fontsize=10, ax=axes.flatten()[index])
    # ajdust borders...
    plt.tight_layout()
    # show the plot
    return plt.show()





# defining a plotting function that we will re-use
def itc_plot_topo_condition(ds, montage, experiment, vmin=None, vmax=None, freq_range=None, savefig=False):
    ''''returns: 
            a plot of the alpha-gamma pac for each condition
      parameters:
          ds: dataset (xarray data_array): The PAC array that you want to plot.
          vmin (int) minimum value to anchor the heatmap.
          vmax (int): maximum value to anchor the heatmap.
          '''
    
    vs = {'experiment1':{'sync_av': slice(1.2,1.8),
                          'audio_only': slice(1.2,1.8),
                          'video_only': slice(1.2,1.8),
                          'async_av': slice(.775,1.2)},
          'experiment2': {'sync_av': slice(1.2,1.8),
                          'audio_only': slice(1.2,1.8),
                          'occlusion_av': slice(1.2,1.8),
                          'async_av': slice(.775,1.2)}}
    
    if not freq_range:
        freqs = slice(np.floor(np.min(ds.frequency.values)),
                      np.ceil(np.max(ds.frequency.values)))
    else: 
        fmin, fmax = freq_range
        freqs = slice(fmin, fmax)
    
     # define the figure canvas..
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8),
                             sharex=True, sharey=True)
    # Loop over conditions..
    conditions = {'experiment1': enumerate(['video_only','audio_only','sync_av','async_av']),
                  'experiment2': enumerate(['occlusion_av','audio_only','sync_av','async_av'])}
    
    
    for index, condition in tqdm(conditions[experiment], total=4, desc='plotting... '): 
        
        # load, prep, and plot data..
        channel_list = zip(ds.channel.values,
                           ds.sel(condition=condition,
                                      time=vs[experiment][condition],
                                      frequency=freqs).mean(['ID',
                                                             'time',
                                                             'frequency']).values)
        
        itc_chans = {chan:value for chan, value in channel_list}
        
        vmin_val = vmin if vmin else np.min(list(itc_chans.values()))
        vmax_val = vmax if vmax else np.max(list(itc_chans.values()))
        
        plot_values_topomap(itc_chans, montage,
                            axes=axes.flatten()[index],
                            vmin=vmin_val,
                            vmax=vmax_val,
                            cmap='viridis')
        
        plt.title(f'{condition}')

    # ajdust borders...
    plt.tight_layout()
    if savefig:
        plt.savefig('topo_itc.png')
        
    # show the plot
    return plt.show()




def compute_cluster_permutation_test(condition_1, condition_2):

    n_perm = 1000  # number of permutations
    tail = 1       # only inspect the upper tail of the distribution
    # perform the correction
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
        [condition_1.values, condition_2.values], n_permutations=n_perm, tail=tail, out_type='indices')

    return clusters, cluster_p_values


def get_cluster_indices(clusters, cluster_p_values, array_to_mask):
    mask = np.zeros_like(array_to_mask.mean('ID').values).astype(bool)
    for pval, cluster in zip(cluster_p_values,clusters):
        if pval <= 0.05:
            for i, j in zip(*cluster):
                mask[i,j] = True
    return mask


'''def plot_sig_clusters(array_to_plot, mask):
    # xlabels = np.round(ds.time[(ds.time >=1.225) & (ds.time <= 1.65)].values)
    # ylabels = np.round(ds.frequency[ds.frequency >= 4].values)
    fig, ax = plt.subplots(1,1, figsize=(6,4)) 
    
    sns.heatmap(array_to_plot.mean('ID').squeeze(), ax=ax)
    plt.gca().invert_yaxis()
    
    sns.heatmap(np.where(mask, array_to_plot.mean('ID').squeeze(), 0),
                ax=ax, alpha=0.3, cbar=False) #cmap='Greys'
    plt.gca().invert_yaxis()
    
    #plt.yticks(ticks=range(len(ylabels)),labels=ylabels, rotation=45)
    #plt.xticks(ticks=range(len(xlabels)),labels=xlabels)
    return plt.show()''';


def plot_sig_clusters(array_to_plot, mask, add_trial_info=False, trial_info_condition=None, trial_info_y_text=None ):
    # xlabels = np.round(ds.time[(ds.time >=1.225) & (ds.time <= 1.65)].values)
    # ylabels = np.round(ds.frequency[ds.frequency >= 4].values)
    fig, ax = plt.subplots(1,1, figsize=(6,4)) 
    
    start_time, end_time = (array_to_plot.time[0].values,
                            array_to_plot.time[-1].values)
    
    first_freq, last_freq = (array_to_plot.frequency[0].values,
                             array_to_plot.frequency[-1].values)
    
    ax.imshow(array_to_plot.mean('ID').squeeze(),
              extent=[start_time, end_time, first_freq, last_freq],
              aspect='auto', origin='lower',
              )
    
    ax.imshow(np.where(mask, array_to_plot.mean('ID').squeeze(), 0),
              extent=[start_time, end_time, first_freq, last_freq],
              aspect='auto', origin='lower', alpha=0.3)
    
    if add_trial_info:
        ucsd.add_trial_information(y_text=18, condition=trial_info_condition, ax=ax, color='w')
        
    return plt.show()




