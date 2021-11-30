# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:44:44 2021

@author: vajra
"""

import os
import scipy
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from .Tint_Matlab import read_cut, getspikes, getpos, centerBox, remBadTrack


# =========================================================================== #
def load_neurons(cut_path: str, tetrode_path: str, channel_no: int) -> tuple: 
    
    '''
        Loads the neuron of interest from a specific cut file. 
        
        Params: 
            cut_path (str): 
                The path of the cut file 
            tetrode_path (str): 
                The path of the tetrode file 
            channel_no (int): 
                The channel number (1,2,3 or 4)

        Returns: 
            Tuple: (channel_data, empty_cell_number) 
            --------
            channel_data (list): 
                Nested list of all the firing data per neuron
            empty_cell_number (int): 
                The 'gap' cell which indicates where the program should stop 
                reading cells from Tint. 
    '''
    
    # Read cut and tetrode data
    cut_data = read_cut(cut_path)
    tetrode_data = getspikes(tetrode_path)
    number_of_neurons = max(cut_data) + 1
    
    # Organize neuron data into list
    channel = [[[] for x in range(2)]for x in range(number_of_neurons)]
    
    for i in range(len(tetrode_data[0])): 
        channel[ cut_data[i] ][0].append(tetrode_data[channel_no][i])
        channel[ cut_data[i] ][1].append(float(tetrode_data[0][i])) 
    
    # Find where there is a break in the neuron data
    # and assign the empty space number as the empty cell
    for i, element in enumerate(channel):
        if (len(element[0]) == 0 or len(element[1]) == 0) and i != 0:
            empty_cell = i
            break
        else:
            empty_cell = i
     
    
    return channel, empty_cell
# =========================================================================== #

def grab_terode_cut_position_files(paths: list) -> tuple:
    
    '''
        Extract tetrode, cut, and position file data /+ a set file

        Params: 
            files (list): 
                List of file paths to tetrode, cut and position file OR folder directory
                containing all files

        Returns: 
            Tuple: tetrode_files, cut_files, pos_files
            --------
            tetrode_files (list): 
                List of all tetrode file paths
            cut_files (list): 
                List of all cut file paths
            pos_files (list): 
                List containing position file paths
    '''

    # Set file references to empty
    pos_files = [] 
    cut_files = [] 
    tetrode_files = []
    
    # Check for set file 
    if len(paths) == 1 and os.path.isdir(paths[0]):
        files = os.listdir(paths[0])
        for file in files:
            if file[-3:] == 'pos': 
                pos_files.append(paths[0] + "/" + file)
            elif file[-1:].isdigit():
                tetrode_files.append(paths[0] + "/" + file)
            elif file[-3:] == 'cut': 
                cut_files.append(paths[0] + "/" + file)
    else: 
        for file in paths: 
            if file[-3:] == 'pos': 
                pos_files.append(file)
            elif file[-1:].isdigit():
                tetrode_files.append(file)
            elif file[-3:] == 'cut': 
                cut_files.append(file)
            else:
                raise NameError("One of the chosen files was not a tetrode, position, or cut file")
    
    if len(pos_files) == 0 or len(cut_files) == 0 or len(tetrode_files) == 0:
        raise NameError("A position, cut, or tetrode file was either not chosen, or not found in the directory")
    

    return tetrode_files, cut_files, pos_files
    
# =========================================================================== #

def get_firing_rate_vs_time(times: np.ndarray, pos_t: np.ndarray, window: int) -> tuple: 

    '''
        Computes firing rate as a function of time

        Params: 
            times (np.ndarray): 
                Array of timestamps of when the neuron fired
            pos_t (np.ndarray):
                Time array of entire experiment 
            window (int): 
                Defines a time window in milliseconds. 

            *Example*
            window: 400 means we will attempt to collect firing data in 'bins' of 
            400 millisecods before computing the firing rates. 

        Returns: 
            tuple: firing_rate, firing_time
            --------
            rate_vector (np.ndarray): 
                Array containing firing rate data across entire experiment
            firing_time: (np.ndarray): 
                Timestamps of when firing occured
        '''
 
    # Initialize zero time elapsed, zero spike events, and zero bin times.
    time_elapsed = 0
    number_of_elements = 1
    bin_time = [0,0]

    # Initialzie empty firing rate and firing time arrays
    firing_rate = [0]
    firing_time = [0]

    # Collect firing data in bins of 400ms
    for i in range(1, len(times)): 
        if time_elapsed == 0: 
            # Set a bin start time
            bin_time_start = times[i-1]

        # Increment time elapsed and spike number as spike event times are iterated over
        time_elapsed += (times[i] - times[i-1])
        number_of_elements += 1

        # If the elapsed time exceeds 400ms
        if time_elapsed > (window/1000): 
            # Set bin end time
            bin_time_end = bin_time_start + time_elapsed
            # Compute rate, and add element to firing_rate array
            firing_rate.append(number_of_elements/time_elapsed)
            firing_time.append( (bin_time_start + bin_time_end)/2 )
            # Reset elapsed time and spiek events number
            time_elapsed = 0
            number_of_elements = 0

    
    
    rate_vector = np.zeros((len(pos_t), 1))
    index_values = []
    for i in range(len(firing_time)):
        index_values.append(  (np.abs(pos_t - firing_time[i])).argmin()  )
        
    firing_rate = np.array(firing_rate).reshape((len(firing_rate), 1))
    rate_vector[index_values] = firing_rate
    firing_rate = np.array(firing_rate).reshape((len(firing_rate), 1))
    rate_vector[index_values] = firing_rate

    return rate_vector, firing_time
  
# =========================================================================== #

def choose_GLM_model(x: np.array, y: np.array, family: str):
    
    # Instantiate models
    poisson_model = sm.GLM(y, x, family=sm.families.Poisson(), missing='drop')
    binomial_model = sm.GLM(y, x, family=sm.families.Binomial(), missing='drop')
    gamma_model = sm.GLM(y, x, family=sm.families.Gamma(), missing='drop')
    gaussian_model = sm.GLM(y, x, family=sm.families.Gaussian(), missing='drop')
    inv_gaussian_model = sm.GLM(y, x, family=sm.families.InverseGaussian(), missing='drop')
    neg_binomial_model = sm.GLM(y, x, family=sm.families.NegativeBinomial(), missing='drop')
    tweedie_model = sm.GLM(y, x, family=sm.families.Tweedie(), missing='drop')
    
    switcher = {
        'Poisson': poisson_model,
        'Binomial': binomial_model,
        'Negative Binomial': neg_binomial_model,
        'Gamma': gamma_model,
        'Gaussian': gaussian_model,
        'Inverse Gaussian': inv_gaussian_model,
        'Tweedie': tweedie_model
    }
    
    return switcher[family]

# =========================================================================== #
    
def grab_position_data(pos_path: str, ppm: int) -> tuple: 
    
    '''
        Extracts position data from .pos file

        Params:
            pos_path (str): 
                Directory of where the position file is stored
            ppm (float): 
                Pixel per meter value 

        Returns: 
            Tuple: pos_x,pos_y,pos_t,(pos_x_width,pos_y_width)
            --------
            pos_x, pos_y, pos_t (np.ndarray): 
                Array of x, y coordinates, and timestamps 
            pos_x_width (float): 
                max - min x coordinate value (arena width)
            pos_y_width (float) 
                max - min y coordinate value (arena length)
    '''

    pos_data = getpos(pos_path, ppm)
    
    # Correcting pos_t data in case of bad position file
    new_pos_t = np.copy(pos_data[2])
    if len(new_pos_t) != len(pos_data[0]): 
        while len(new_pos_t) != len(pos_data[0]):
            new_pos_t = np.append(new_pos_t, float(new_pos_t[-1] + 0.02))
    
    Fs_pos = pos_data[3]
    
    pos_x = pos_data[0]
    pos_y = pos_data[1]
    pos_t = new_pos_t
    
    # Rescale coordinate values with respect to a center point
    # (i.e arena center = origin (0,0))
    center = centerBox(pos_x, pos_y)
    pos_x = pos_x - center[0]
    pos_y = pos_y - center[1]
    
    # Correct for bad tracking
    pos_data_corrected = remBadTrack(pos_x, pos_y, pos_t, 2)
    pos_x = pos_data_corrected[0]
    pos_y = pos_data_corrected[1]
    pos_t = pos_data_corrected[2]  
    
    # Remove NaN values 
    nonNanValues = np.where(np.isnan(pos_x) == False)[0]
    pos_t = pos_t[nonNanValues]
    pos_x = pos_x[nonNanValues]
    pos_y = pos_y[nonNanValues]
    
    # Smooth data using boxcar convolution
    B = np.ones((int(np.ceil(0.4 * Fs_pos)), 1)) / np.ceil(0.4 * Fs_pos)
    pos_x = scipy.ndimage.convolve(pos_x, B, mode='nearest')
    pos_y = scipy.ndimage.convolve(pos_y, B, mode='nearest')
    
    pos_x_width = max(pos_x) - min(pos_x)
    pos_y_width = max(pos_y) - min(pos_y)
    
    return pos_x, pos_y, pos_t, (pos_x_width, pos_y_width)