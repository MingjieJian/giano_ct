#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import rulerwd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from scipy.optimize import minimize
from scipy.signal import medfilt

from telfit import Modeler
import os, sys
import shutil

import spectres

import argparse

def find_overlap_range(arr1, arr2):
    # 找出两个数组中的最小值和最大值
    min_val = max(min(arr1), min(arr2))
    max_val = min(max(arr1), max(arr2))
    
    # 计算重叠范围
    if max_val >= min_val:
        overlap_range = (min_val, max_val)
    else:
        overlap_range = None
    
    return overlap_range

def afs(spec, q=0.95, d=0.25, input_flux='flux'):
    
    spec_r = spec[['wave', input_flux]]
    spec_r.columns = ['wv', 'intens']
    spec_r = pandas2ri.py2rpy_pandasdataframe(spec_r)

    q_r = robjects.FloatVector([q])
    d_r = robjects.FloatVector([d])

    AFS = robjects.r['AFS']
    result = AFS(spec_r, q=q_r, d=d_r)
    
    return result

def split_continuous_elements(arr):
    result = []
    temp = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            temp.append(arr[i])
        else:
            result.append(temp)
            temp = [arr[i]]
    
    result.append(temp)
    
    return result

def tell_correct_std(x, obs_y, tell_y):
    std = np.std(obs_y / (1 - (1-tell_y)*x))
    return std

def replace_continuous_true(boo, threshold=40):
    indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
    for i in range(len(indices)-1):
        if indices[i+1] - indices[i] >= threshold:
            boo[indices[i]:indices[i+1]] = False
    return boo

def std_running_filter(data, window_size):
    """
    使用标准差进行运行滤波
    
    参数:
        - data: 一维数据数组
        - window_size: 窗口大小
    返回值:
        - filtered_data: 运行标准差滤波后的数据
    """
    n = len(data)
    filtered_data = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window_data = data[start:end]
        # Perform sigma clipping
        len_old = 1
        len_new = len(window_data[np.isnan(window_data)])
        while len_old != len_new:
            len_old = len(window_data[np.isnan(window_data)])
            window_data[np.abs(window_data - np.nanmedian(window_data)) > 2.5*np.nanstd(window_data)] = np.nan
            len_new = len(window_data[np.isnan(window_data)])
        filtered_data[i] = np.nanstd(window_data)
    return filtered_data

# Load the content in the R file
with open('AFS/functions/AFS.R', 'r') as f:
    custom_code = f.read()
# Load the R function 
robjects.r(custom_code)
AFS = robjects.r['AFS']

def raw_continuum(spec, std_upscale=3, filter_window=101):
    '''
    Perform raw continuum normalizaiton.
    '''

    order_list = spec.groupby('order').size().index

    # AFS continuum
    print('****************************')
    print('Performing raw continuum fitting')
    for order in tqdm(order_list):
        indices = spec['order'] == order
        afs_result = afs(spec[indices])
        spec.loc[indices, 'flux_con_raw'] = afs_result
        raw_cont = spec.loc[indices, 'flux'] / spec.loc[indices, 'flux_con_raw']
        
        spec.loc[indices, 'flux_median_filtered'] = medfilt(spec.loc[indices, 'flux'], kernel_size=filter_window)
        spec.loc[indices, 'flux_std_filtered'] = std_upscale * std_running_filter(spec.loc[indices, 'flux'].values, window_size=filter_window)
        std_diff = np.abs(np.diff(spec.loc[indices, 'flux_median_filtered'] + spec.loc[indices, 'flux_std_filtered']))
        std_diff = np.concatenate([std_diff[0:1], std_diff])
        
        indices_up = raw_cont < spec.loc[indices, 'flux_median_filtered']
        spec.loc[indices & indices_up, 'flux_con_raw'] = spec.loc[indices & indices_up, 'flux_median_filtered']
        
    spec['flux_con_input'] = spec['flux']
    indices_peak = ((spec['flux'] > spec['flux_median_filtered'] + spec['flux_std_filtered']) | (spec['flux'] > spec['flux'] / spec['flux_con_raw'])).values
    spec['indices_con_remove'] = replace_continuous_true(indices_peak)

    for ele in split_continuous_elements(spec[spec['indices_con_remove']].index):
        indices_inperp = [ele[0]-1] + ele + [ele[-1]+1]
        indices_edge = [ele[0]-1] + [ele[-1]+1]
        spec.loc[indices_inperp, 'flux_con_input'] = np.interp(spec.loc[indices_inperp, 'wave'], spec.loc[indices_edge, 'wave'], spec.loc[indices_edge, 'flux'])
        spec.loc[indices_inperp, 'flux_con_input'] = spec.loc[indices_inperp, 'flux_con_input'] * (1+np.random.randn(len(spec.loc[indices_inperp, 'flux_con_input']))*0.01)

    # Second continuum estimation
    for order in tqdm(order_list):
        indices = spec['order'] == order
        afs_result = afs(spec[indices], input_flux='flux_con_input')
        spec.loc[indices, 'flux_con_final'] = spec.loc[indices, 'flux'] / (spec.loc[indices, 'flux_con_input'] / afs_result)
        
        # Retrive the telluric continuum
        if order in tell_correct_type['tell_con']:
            indices_tell_model = (standard_telluric_spectra['wave'] >= np.min(spec.loc[indices, 'wave']) - 1) & (standard_telluric_spectra['wave'] <= np.max(spec.loc[indices, 'wave']) + 1)
            spec_tell_model = pd.DataFrame([spec.loc[indices, 'wave'].values, 
                                            spectres.spectres(spec.loc[indices, 'wave'].values, standard_telluric_spectra['wave'][indices_tell_model], standard_telluric_spectra['flux'][indices_tell_model])]).T
            spec_tell_model.columns = ['wave', 'flux_tell_model']
            
            afs_result_tell_model = afs(spec_tell_model, input_flux='flux_tell_model')
            spec.loc[indices, 'flux_con_final'] = spec.loc[indices, 'flux_con_final'].values * spec_tell_model['flux_tell_model'].values / afs_result_tell_model

    return spec

def telluric_correction(spec, input_spec):

    order_list = spec.groupby('order').size().index

    try:
        file = fits.open(f'{input_spec[:-4]}.fits')
        header = file[0].header
        EL = np.rad2deg(header['EL'])
    except FileNotFoundError:
        EL = 50

    spec['tell_flux'] = spec['flux_con_final']
    spec['flux_con_notell'] = spec['flux_con_final']

    for order in tqdm(order_list):
    # for order in tqdm([77]):
        indices = spec['order'] == order
        if order in tell_correct_type['None']:
            # Skip the telluric correction since the telluric lines are too strong.
            spec.loc[indices, 'tell_flux'] = np.nan
            spec.loc[indices, 'flux_con_notell'] = 1
        else:
            res = rulerwd.telfit_wrapper.Telfit(spec.loc[indices, 'wave'].values, spec.loc[indices, 'flux_con_final'].values, 
                                                EL, 285, 'TNG', 50000, air_wave=False, suppress_stdout=True)
            res.y += 1e-3
            tell_indices = 1 - res.y > 0.01
            if order in tell_correct_type['tell_con']:
                cut_wav = tell_correct_type['tell_con_wav'][list(tell_correct_type['tell_con']).index(order)]
                if cut_wav > 0:
                    tell_indices = tell_indices & (res.x*10 <= cut_wav)
                    print('Order {}, cut_wav = {}'.format(order, cut_wav))
                elif cut_wav < 0:
                    tell_indices = tell_indices & (res.x*10 >= -cut_wav)
                    print('Order {}, cut_wav = {}'.format(order, cut_wav))

            tell_ratio = minimize(tell_correct_std, 1, 
                                args=(spec.loc[indices, 'flux_con_final'][tell_indices].values, res.y[tell_indices]),
                                bounds=[(0.5, 1.5)])['x'][0]
            spec.loc[indices, 'tell_flux'] = (1 - (1-res.y) * tell_ratio)
            spec.loc[indices, 'flux_con_notell'] = spec.loc[indices, 'flux_con_final'] / spec.loc[indices, 'tell_flux']
    return spec
    
def combine(spec, edge_percent=0.2):

    order_list = spec.groupby('order').size().index

    spec_all = spec[spec['wave'] < 0]
    wav_overlap_all = []

    i = 1
    for order in order_list[::-1]:
        
        if order == 32:
            spec_all = pd.concat([spec_all, spec[(spec['order'] == order)]])
        else:
            wav_overlap_start = spec.loc[(spec['order'] == order-1), 'wave'].iloc[0] + 1
            wav_overlap_end = spec.loc[(spec['order'] == order), 'wave'].iloc[-1] - 1
            if wav_overlap_end <= wav_overlap_start:
                wav_overlap_mid = np.nan
            else:
                wav_overlap_mid = np.mean([wav_overlap_start, wav_overlap_end])
            wav_overlap_all.append([wav_overlap_start, wav_overlap_mid, wav_overlap_end])

            if wav_overlap_end <= wav_overlap_start:
                spec_all = pd.concat([spec_all, spec[(spec['order'] == order)]])
            else:
                # Combining the spectra
                if order == 81:
                    spec_all = pd.concat([spec_all, spec[(spec['order'] == order) & (spec['wave'] < wav_overlap_start)]])
                else:
                    spec_all = pd.concat([spec_all, spec[(spec['order'] == order) & (spec['wave'] >= wav_overlap_all[-2][-1]) & (spec['wave'] < wav_overlap_start)]])

                # First half overlap
                spec_t1 = spec[(spec['order'] == order) & (spec['wave'] >= wav_overlap_start) & (spec['wave'] < wav_overlap_mid)]
                spec_t2 = spec[(spec['order'] == order-1) & (spec['wave'] >= wav_overlap_start-0.5) & (spec['wave'] < wav_overlap_mid+0.5)]
                spec_t2_resample = spec_t1[spec_t1.columns]

                for col in ['flux', 'snr', 'flux_con_raw', 'flux_median_filtered', 'flux_std_filtered', 'flux_con_input', 'flux_con_final',
                            'tell_flux', 'flux_con_notell']:
                    spec_t2_resample[col] = spectres.spectres(spec_t1['wave'].values, spec_t2['wave'].values, spec_t2[col].values)

                edge_length = int(len(spec_t2_resample) * 0.2)
                snr_weights = np.concatenate([np.interp(np.arange(edge_length), [0, edge_length], [0.01, 1]), np.ones(len(spec_t2_resample) - edge_length)])
                spec_t2_resample['snr'] = spec_t2_resample['snr'] * snr_weights

                spec_t1_add = spec_t2_resample[spec_t2_resample.columns]
                spec_t1_add['snr'] = np.sqrt(spec_t1['snr']**2 + spec_t2_resample['snr']**2)
                
                for col in ['flux', 'flux_median_filtered', 'flux_std_filtered', 'flux_con_input']:
                    spec_t1_add[col] = spec_t1[col] + spec_t2_resample[col]
                for col in ['flux_con_raw', 'flux_con_final', 'tell_flux', 'flux_con_notell']:
                    spec_t1_add[col] = np.average([spec_t1[col], spec_t2_resample[col]], axis=0,
                                                        weights=[spec_t1['snr']**2, spec_t2_resample['snr']**2])

                spec_all = pd.concat([spec_all, spec_t1_add])

                # Second half overlap
                spec_t1 = spec[(spec['order'] == order) & (spec['wave'] >= wav_overlap_mid-0.5) & (spec['wave'] < wav_overlap_end+0.5)]
                spec_t2 = spec[(spec['order'] == order-1) & (spec['wave'] >= wav_overlap_mid) & (spec['wave'] < wav_overlap_end)]
                spec_t1_resample = spec_t2[spec_t2.columns]

                for col in ['flux', 'snr', 'flux_con_raw', 'flux_median_filtered', 'flux_std_filtered', 'flux_con_input', 'flux_con_final',
                            'tell_flux', 'flux_con_notell']:
                    spec_t1_resample[col] = spectres.spectres(spec_t2['wave'].values, spec_t1['wave'].values, spec_t1[col].values)

                ten_percent_length = int(len(spec_t1_resample) * edge_percent)
                snr_weights = np.concatenate([np.ones(len(spec_t1_resample) - ten_percent_length), np.interp(np.arange(ten_percent_length), [0, ten_percent_length], [1, 0.01])])
                spec_t1_resample['snr'] = spec_t1_resample['snr'] * snr_weights

                spec_t2_add = spec_t1_resample[spec_t1_resample.columns]
                spec_t2_add['snr'] = np.sqrt(spec_t2['snr']**2 + spec_t1_resample['snr']**2)
                
                for col in ['flux', 'flux_median_filtered', 'flux_std_filtered', 'flux_con_input']:
                    spec_t2_add[col] = spec_t2[col] + spec_t1_resample[col]                
                for col in ['flux_con_raw', 'flux_con_final', 'tell_flux', 'flux_con_notell']:
                    spec_t2_add[col] = np.average([spec_t2[col], spec_t1_resample[col]], axis=0,
                                                        weights=[spec_t2['snr']**2, spec_t1_resample['snr']**2])
                    
                spec_all = pd.concat([spec_all, spec_t2_add])

    spec_all = spec_all.drop(columns='indices_con_remove')
    return spec_all
    
def plot_result(spec, spec_all, output_folder, spike_rej=False, cont_nor=False, tell_corr=False, combine=False, final=True):
    '''
    Plot the raw continuum spectra.
    '''

    order_list = spec.groupby('order').size().index

    if spike_rej:
        for order in order_list[::-1]:
            plt.figure(figsize=(13, 3), dpi=150)

            indices = spec['order'] == order
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux'], lw=1, alpha=0.5, label='Raw spectra')
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_input'], lw=1, zorder=0, label='Raw spectra (spikes removed)')
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux'] / spec.loc[indices, 'flux_con_raw'], lw=1, label='Initial continuum')
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_median_filtered'], '--', lw=1, zorder=0, label='Medain filtered spectra')
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_median_filtered'] + spec.loc[indices, 'flux_std_filtered'], 
                    lw=1, zorder=0, label='Std filtered spectra')            
            indices_removed = indices & spec['indices_con_remove']
            plt.plot(spec.loc[indices_removed, 'wave'], spec.loc[indices_removed, 'flux'], 
                    'x', color='red', markersize=4, alpha=0.5, label='Removed spikes')
            
            plt.title('Order {}'.format(order))
            plt.legend(fontsize=7)
            plt.tight_layout()
            
            plt.savefig(output_folder + 'spike_rej/order{}.pdf'.format(order))
            plt.close()

    if cont_nor:
        for order in tqdm(order_list[::-1]):
            plt.figure(figsize=(13, 3), dpi=100)

            indices = spec['order'] == order
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux'] / np.max(spec.loc[indices, 'flux']), 
                    lw=1, alpha=0.4, color='gray', label='Raw spectra')
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_final'], lw=1, label='Normalized spectra')
            xlim = plt.xlim()
            plt.xlim(xlim)

            indices = spec['order'] == order - 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux'] / np.max(spec.loc[indices, 'flux']), 
                    c='gray', lw=1, zorder=0, alpha=0.6, ls='-.', label='Raw spectra (previous order)')
            indices = spec['order'] == order + 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux'] / np.max(spec.loc[indices, 'flux']), 
                    c='gray', lw=1, zorder=0, alpha=0.6, ls='--', label='Raw spectra (next order)')
            plt.title('Order {}'.format(order))
            plt.axhline(1, ls='--', color='brown', alpha=0.5)
            
            indices = (standard_telluric_spectra['wave'] >= xlim[0]) & (standard_telluric_spectra['wave'] <= xlim[1])
            plt.plot(standard_telluric_spectra.loc[indices, 'wave'], standard_telluric_spectra.loc[indices, 'flux'], lw=1, alpha=0.4, label='Standard Telluric spectra')
            
            plt.legend(fontsize=7)
            plt.tight_layout()
            
            plt.savefig(output_folder + 'cont_nor/order{}.pdf'.format(order))
            plt.close()

    if tell_corr:
        for order in tqdm(order_list):
            plt.figure(figsize=(13, 3*2))

            indices = spec['order'] == order
            plt.subplot(211)
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_final'], label='Normalized spectra')
            plt.scatter(spec.loc[indices & spec['indices_con_remove'], 'wave'], 
                        spec.loc[indices & spec['indices_con_remove'], 'flux_con_final'],
                        marker='x', color='red', label='Removed spikes')

            if order not in tell_correct_type['None']:
                plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'tell_flux'], alpha=0.5, label='Fitted telluric spectra')
            plt.axhline(1, ls='--', color='brown', alpha=0.5)
            
            plt.title('Order {}'.format(order))
            plt.legend(fontsize=7)
            ylim = plt.ylim()

            plt.subplot(212)

            spec['flux_con_notell_input'] = spec['flux_con_notell']
            spec['flux_con_notell_con'] = spec['flux_con_notell']
            spec['indices_tell_con_remove'] = False

            if order not in tell_correct_type['None']:

                spec.loc[indices, 'indices_tell_con_remove'] = spec.loc[indices, 'flux_con_notell'] > np.median(spec.loc[indices, 'flux_con_notell']) + np.nanstd(spec.loc[indices, 'flux_con_notell'])
                for ele in split_continuous_elements(spec[indices & ((spec['indices_con_remove']) | (spec['indices_tell_con_remove']))].index):
                    indices_inperp = [ele[0]-1] + ele + [ele[-1]+1]
                    indices_edge = [ele[0]-1] + [ele[-1]+1]
                    spec.loc[indices_inperp, 'flux_con_notell_input'] = np.interp(spec.loc[indices_inperp, 'wave'], spec.loc[indices_edge, 'wave'], spec.loc[indices_edge, 'flux_con_notell'])
                    spec.loc[indices_inperp, 'flux_con_notell_input'] = spec.loc[indices_inperp, 'flux_con_notell_input'] * (1+np.random.randn(len(spec.loc[indices_inperp, 'flux_con_notell_input']))*0.05)

            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], 
                    zorder=1, label='Telluric corrected spectra')

            plt.ylim(ylim)

            plt.axhline(1, ls='--', color='brown', alpha=0.5, zorder=0)
            
            plt.legend(fontsize=7)
            plt.tight_layout()
            
            plt.savefig(output_folder + 'tell_corr/order{}.pdf'.format(order))
            plt.close()

    if combine:
        wav_overlap_all = []

        i = 1
        for order in order_list[::-1]:
            plt.figure(figsize=(13, 3), dpi=100)

            indices = spec['order'] == order
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], lw=1)
            plt.xlim(plt.xlim())
            
            if order == 32:
                spec_all = pd.concat([spec_all, spec[(spec['order'] == order)]])
            else:
                wav_overlap_start = spec.loc[(spec['order'] == order-1), 'wave'].iloc[0] + 1
                wav_overlap_end = spec.loc[(spec['order'] == order), 'wave'].iloc[-1] - 1
                if wav_overlap_end <= wav_overlap_start:
                    wav_overlap_mid = np.nan
                else:
                    wav_overlap_mid = np.mean([wav_overlap_start, wav_overlap_end])
                wav_overlap_all.append([wav_overlap_start, wav_overlap_mid, wav_overlap_end])

            # Plot
            indices = spec['order'] == order - 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], 
                    c='gray', lw=1, zorder=0, alpha=0.6)
            indices = spec['order'] == order + 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], 
                    c='gray', lw=1, zorder=0, alpha=0.6)
            plt.title(order)
            plt.axhline(1, ls='--', color='brown')

            plt.axvspan(wav_overlap_start, wav_overlap_mid, alpha=0.2)
            plt.axvspan(wav_overlap_mid, wav_overlap_end, alpha=0.2, color='gray')
            if order != 81:
                plt.axvspan(wav_overlap_all[-2][1], wav_overlap_all[-2][2], alpha=0.2)
                plt.axvspan(wav_overlap_all[-2][0], wav_overlap_all[-2][1], alpha=0.2, color='gray')
            
            i += 1
            plt.ylim(0, 1.15)
            
            plt.tight_layout()
            plt.savefig(f'{output_folder}/combine/order{order}.pdf')
            plt.close()

    if final:
        for order in order_list[::-1]:
            plt.figure(figsize=(13, 3))

            indices = spec['order'] == order
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], lw=1, c='gray', alpha=0.6, zorder=0)
            plt.xlim(plt.xlim())
            plt.plot(spec_all['wave'], spec_all['flux_con_notell'], lw=1, zorder=5)

            indices = spec['order'] == order - 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], 
                    c='gray', lw=1, zorder=0, alpha=0.6, ls='-.')
            indices = spec['order'] == order + 1
            plt.plot(spec.loc[indices, 'wave'], spec.loc[indices, 'flux_con_notell'], 
                    c='gray', lw=1, zorder=0, alpha=0.6, ls='--')
            plt.title(f'Order {order}')
            plt.axhline(1, ls='--', color='brown')
 
            plt.ylim(0, 1.15)
            
            plt.tight_layout()
            plt.savefig(f'{output_folder}/final/order{order}.pdf')
            plt.close()

            plt.figure(figsize=(13, 3))
            plt.plot(spec_all['wave'], spec_all['flux_con_notell'], lw=0.1)
            plt.ylim(0, 1.15)

            plt.tight_layout()
            plt.savefig(f'{output_folder}/final/all_short.pdf')
            plt.close()

def generate_report(input_spec, output_folder):
    input_spec_report = input_spec.replace('_', '\_')
    report_replact_dict = {
        'input_file_name': f'\\texttt{{{input_spec_report}}}',
        'EL': '50', 'std_upscale': '3', 'filter_window': '61',
        'spike_rej': '\\checkmark', 'cont_nor': '\\checkmark', 'tell_corr': '\\checkmark', 'combine': '\\checkmark'
    }

    # Copy the template to output folder
    _ = shutil.copy('report_template/aa.bst', f'{output_folder}/report')
    _ = shutil.copy('report_template/refs.bib', f'{output_folder}/report')
    report_content = open('report_template/report.tex', 'r').readlines()
    report_content = '\t'.join(report_content)

    for key in report_replact_dict.keys():
        report_content = report_content.replace(f'[{key}]', report_replact_dict[key])

    with open(f'{output_folder}/report/report.tex', 'w') as file:
        file.writelines(report_content.split('\t'))

    working_folder = os.getcwd()
    os.chdir(f'{output_folder}/report/')
    os.system(f'pdflatex -interaction=nonstopmode -quiet report')
    os.system(f'bibtex report')
    os.system(f'pdflatex -interaction=nonstopmode -quiet report')
    os.system(f'pdflatex -interaction=nonstopmode -quiet report')
    os.chdir(working_folder)

# Read the standard telluric spectra
standard_telluric_spectra = np.load('standard_telluric_spectra.npy')
standard_telluric_spectra = {'wave':standard_telluric_spectra[0, :], 'flux':standard_telluric_spectra[1, :]}

tell_correct_type = {
    'usual': [
        81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70,
        65, 64, 63, 62, 61, 60, 59,
        50, 49, 48, 47, 46, 45, 44,
        36, 35, 34, 33
    ],
    'tell_con': [
        69, 68, 67, 66, 58, 57, 53, 52, 51, 43, 42, 41, 40, 39, 38, 37, 32
    ],
    'tell_con_wav': [
        0, 0, 0, 0, 0, 13425, -14550, 0, 0, 17900, 0, 0, 0, -19700, 0, 0, 0
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    'None': [
        56, 55, 54,
        42, 41, 40
    ]
}

def main(input_spec, output_folder):
    print(input_spec, output_folder)

    # Create the result folder
    for folder in [output_folder, output_folder+'spike_rej', output_folder+'cont_nor', output_folder+'tell_corr', output_folder+'combine', output_folder+'final', output_folder+'report']:
        os.makedirs(folder, exist_ok=True)

    # sys.stdout = open(output_folder + 'giano_ct.log', 'w')

    # Read the specta
    spec = pd.read_csv(f'{input_spec}', sep=' +', engine='python', skiprows=23, names=['order', 'wave', 'flux', 'snr'])
    spec['wave'] *= 10
    # Preprocessing, avoid numeratic errors: 
    spec = spec[spec['wave'] > 9375]
    spec.loc[spec['snr'] < 0.01, 'snr'] = 0.01
    spec.loc[spec['flux'] < 0.01, 'flux'] = 0.01
    
    spec = raw_continuum(spec)
    spec = telluric_correction(spec, input_spec)
    spec_all = combine(spec)
    plot_result(spec, spec_all, output_folder)

    # Save the spectra
    spec.to_csv(f'{output_folder}/spec_con_tell.csv', index=False)
    spec_all.to_csv(f'{output_folder}/spec_con_tell_combine.csv', index=False)

    generate_report(input_spec, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIANO-CT pipeline")

    # 添加输入参数
    parser.add_argument("input_spec", help="The input spectra")
    parser.add_argument("output_folder", help="The output folder")

    # 添加带有默认值的输入参数
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode", default=False)

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.input_spec, args.output_folder)