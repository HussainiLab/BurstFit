# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:40:32 2021

@author: vajra
"""

from PyQt5.QtCore import QObject, pyqtSignal

# =========================================================================== #    

class WorkerSignals(QObject):
    '''
        Defines the signals available from a running worker thread.
    
        Supported signals are:
    
        error
            Invoke when error occurs
    
        return_data
            tuple of return data from worker
    
        progress
            Progress indicator for worker thread
    '''
    
    error = pyqtSignal(str)
    return_data = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    
# =========================================================================== # 