# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:49:58 2021

@author: vajra
"""
import numpy as np
from matplotlib import pyplot as plt
from functions.neuron_functions import *
from functions.Tint_Matlab import speed2D

# =========================================================================== #  
def compute_GLM(self, files, cell, ppm, family: str, graph: str, **kwargs):
    
    # Instantiate error message as None
    error = None
    
    # Grab position, tetrode and cut file
    pos_file = files[0]
    cut_file = files[1]
    tetrode_file = files[2]
    
    # If kwarg exists, no need to repeat computation
    cell_data = kwargs.get('cell_data', None)
    if cell_data == None: 
        # Loading raw spike data from tetrode 
        raw_spike_data, empty_cell = load_neurons(cut_file, tetrode_file, channel_no=1)
        final_cell = len(raw_spike_data) - 1
        # Grab position data
        pos_x, pos_y, pos_t, arena_size = grab_position_data(pos_file, ppm)
        cell_data = (raw_spike_data, (pos_x, pos_y, pos_t, arena_size), final_cell)
    else:
        raw_spike_data = cell_data[0]
        pos_x, pos_y, pos_t, arena_size = cell_data[1][0], cell_data[1][1], cell_data[1][2], cell_data[1][3]
        final_cell = cell_data[2]
    
    # Emit 25% progress done for progress bar
    self.signals.progress.emit(25)
                
    # Load neuron data and organize
    unit_data = raw_spike_data[cell][1]
    firing_data, firing_time = get_firing_rate_vs_time(unit_data, pos_t, 400)
    speed = speed2D(pos_x, pos_y, pos_t)
    
    self.signals.progress.emit(50)
    
    # Instantiate endogenous and exogenous variables 
    y1 = firing_data.flatten()
    x1 = pos_t.flatten()
    y2 = speed.flatten()
    x2 = pos_t.flatten()
    
    # Determine what model to predict with based on user choice
    model_rate = choose_GLM_model(x1, y1, family)
    model_rate_and_speed = choose_GLM_model(x2, y2, family)
    
    self.signals.progress.emit(75)
    
    # Predict 
    
    
    if graph is 'Rate':
        try:
            
            predictor = model_rate.fit()
            prediction = predictor.predict(x1)
            
        except Exception as e:
            
            
            error = str(e)
            self.signals.error.emit(error)
            return
        
        self.signals.progress.emit(100)
        return cell_data, x1, y1, prediction
    
    else:
        try:
            
            predictor = model_rate_and_speed.fit()
            prediction = predictor_rate.predict(x2)
            
        except Exception as e:
            
            error = str(e)
            self.signals.error.emit(error)
            return
        
        self.signals.progress.emit(100)
        return cell_data, x2, y2, prediction
    