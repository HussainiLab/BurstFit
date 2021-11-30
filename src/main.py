# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:53:36 2021
@author: vajramsrujan
"""

import os
import sys
import xlwings as xw
import numpy as np
import matplotlib
import shutil

from compute_GLM import compute_GLM
from worker_thread.Worker import Worker
from openpyxl.utils.cell import get_column_letter
from PIL import Image, ImageQt
from functools import partial
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, Qt, QThreadPool
from PyQt5.QtWidgets import *

matplotlib.use('Qt5Agg')

# =========================================================================== #

class MplCanvas(FigureCanvasQTAgg):

    '''
        Canvas class to generate matplotlib plot widgets in the GUI.
        This class takes care of plotting.
    '''

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
         
# =========================================================================== #

class mainWindow(QWidget):
    
    '''
        Class creates an interactive window for plotting firing data regression models.
    '''

    def __init__(self):
        
        QWidget.__init__(self)
        
        # Setting main window geometry
        self.setGeometry(50, 50, 1000, 900)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.center()
        self.mainUI()
        
    # ------------------------------------------- # 
    
    def center(self):

        '''
            Centers the GUI window on screen upon launch
        '''

        # geometry of the main window
        qr = self.frameGeometry()

        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())
        
    # ------------------------------------------- #  
    def mainUI(self):
        
        '''
            Centers the GUI window on screen upon launch
        '''

        # Initialize layout and title
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Cell viewer")
        
        # Data
        self.x = None
        self.y = None
        self.prediction = None              # Holds GLM prediction
        self.ppm = None
        self.files = [None, None, None]           # Holds a reference to pos, cut and tetrode file
        self.active_folder = ''             # Holds last directory path opened by user for choosing files
        self.cell = 1                       # Keeps track of current selected neuron
        self.cell_data = None               # Holds spike data 
        self.re_render = False              # Determines if the list widget reference to neurons needs to be reloaded 
        self.family = 'Poisson'
        self.error = None
        self.graphType = 'Rate'
         
        # Widget creation
        session_Label = QLabel("Session:")
        model_Label = QLabel("Model type:")
        graph_Label = QLabel("Graph type:")
        ppm_Label = QLabel("Pixel Per Meter value:")
        ppmTextBox = QLineEdit(self)
        self.session_Text = QLabel()
        self.neuron_Label = QLabel("Available Neurons")
        self.modelBox = QComboBox()
        self.graphBox = QComboBox()
        
        quit_button = QPushButton('Quit', self)
        browse_button = QPushButton('Browse files', self)
        save_button = QPushButton('Save images', self)
        self.bar = QProgressBar(self)
        self.listWidget = QListWidget()
        
        # Add items to combobox
        families = ['Poisson', 'Binomial', 'Negative Binomial', 'Gamma', 'Gaussian', 
                    'Inverse Gaussian', 'Tweedie']
        for family in families:
            self.modelBox.addItem(family)
        
        self.graphBox.addItem("Rate")
        self.graphBox.addItem("Rate_vs_Speed")
        
        # Create canvas widgets used later for plotting image data
        self.rate_plot = MplCanvas()
        
        # Instantiating widget properties 
        self.bar.setOrientation(Qt.Vertical)
        self.modelBox.setFixedWidth(125)
        self.graphBox.setFixedWidth(125)
        self.listWidget.setFixedWidth(100)
        browse_button.setFixedWidth(300)
        ppmTextBox.setFixedWidth(125)
        
        # Placing widgets
        self.layout.addWidget(browse_button, 0, 1)
        self.layout.addWidget(quit_button, 0, 2)
        self.layout.addWidget(graph_Label, 1,0)
        self.layout.addWidget(self.graphBox, 1,1)
        self.layout.addWidget(save_button, 1,2)
        self.layout.addWidget(model_Label, 2, 0)
        self.layout.addWidget(self.modelBox, 2, 1)
        self.layout.addWidget(ppm_Label, 3, 0)
        self.layout.addWidget(ppmTextBox, 3, 1)
        self.layout.addWidget(self.neuron_Label, 4, 0)
        self.layout.addWidget(self.listWidget, 5, 0)
        self.layout.addWidget(self.rate_plot, 5, 1)
        self.layout.addWidget(self.bar, 5, 2)
        
        # Widget signaling
        quit_button.clicked.connect(self.quitClicked)
        browse_button.clicked.connect(self.runSession)
        self.listWidget.currentItemChanged.connect(self.cellChanged)
        self.graphBox.activated[str].connect(self.graphChanged)
        self.modelBox.activated[str].connect(self.modelChanged)
        ppmTextBox.textChanged[str].connect(partial(self.textBoxChanged, 'ppm'))
                
    # ------------------------------------------- #  
    
    def cellChanged(self, current, previous): 

        '''
            Detect if the user chooses a different neuron
        '''

        if self.listWidget.currentItem is not None and current is not None:
            self.cell = int(current.text())
            self.runWorkerThread()
      
    # ------------------------------------------- #  
    
    def modelChanged(self, value): 

        '''
            Updates GLM model type and re-runs the computation.
        '''

        self.family = value
        if self.cell_data is not None:
            self.runWorkerThread()
        
    # ------------------------------------------- #  
    
    def graphChanged(self, value): 

        '''
            Updates graph type between rate graph and rate vs speed graph
        '''
        
        self.graphType = value
        if self.cell_data is not None:
            self.runWorkerThread()
        
    # ------------------------------------------- #  
    
    def textBoxChanged(self, label):
        
        '''
            Checks and sets pixel per meter field. 
        '''

        cbutton = self.sender()
        curr_string = str(cbutton.text()).split(',')
        
        if label == 'ppm': 
            if len(curr_string) == 1 and len(curr_string[0]) == 0:
                self.ppm = None
            if curr_string[0].isnumeric():
                self.ppm = int(curr_string[0])
                
     # ------------------------------------------- # 
     
    def openFileNamesDialog(self):
    
        '''
            Open file dialog to select files.
        '''

        # Prepare error dialog window 
        self.error_dialog = QErrorMessage()
        
        # Error check prior to worker thread launch
        if self.ppm == None:
            self.error_dialog.showMessage('ppm field is blank, or is non-numeric. Please enter an appropriate number.')
            return
        elif self.ppm < 0: 
            self.error_dialog.showMessage('ppm must be greater than zero')
            return
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Choose .pos file, cut file and tetrode file", self.active_folder, options=options)
        
        if len(files) > 0:
            self.active_folder = dir_path = os.path.dirname(os.path.realpath((files[0]))) 
        
        for file in files:
            extension = file.split(sep='.')[1]
            if 'pos' in extension:
                self.files[0] = file
            elif 'cut' in extension:
                self.files[1] = file
            elif extension.isnumeric():
                self.files[2] = file
            else:    
                self.error_dialog.showMessage('You must choose one .pos one .cut, and one tetrode file')
                return
        
        self.cell = 1
        self.session_Text.setText(str(self.files[0]))
       
        # Sets render flag to re render list of neurons
        self.re_render = True
        # Invokes worker thread function
        self.runWorkerThread()
        
    # ------------------------------------------- #
    
    def quitClicked(self):

        '''
            Application exit
        '''

        print('quit')
        QApplication.quit()
        self.close() 
                   
    # ------------------------------------------- #  
    
    def progressBar(self, n):

        '''
            Creates a progress bar

            Params:
                n (int):
                    Sets the number of possible states in the progress bar

            Returns:
                No return
        '''

        n = int(n)

        # setting geometry to progress bar
        self.bar.setValue(n)
        
    # ------------------------------------------- #  
    
    def updateLabel(self, value): 

        '''
            Update the progress bar label based on
            worker thread completion.

            Params:
                value (int): 
                    Meant to reflect worker thread progress in percent value

                    Returns:
                        No return
        '''

        self.progressBar_Label.setText(value)
        
    # ------------------------------------------- #  
    
    def errorOccured(self, error):
        
        # Prepare error dialog window 
        self.error_dialog = QErrorMessage()
        
        # Show error
        self.error_dialog.showMessage(error)
        
    # ------------------------------------------- #  
    
    def runSession(self):
        
        '''
            Resets variables and invokes worker thread function.
        '''
        
        # Clears any existing cell data reference
        self.cell_data = None
        # Querys user for appropriate file selection
        self.openFileNamesDialog()
        
    # ------------------------------------------- #       
       
    def setData(self, data):
        
        '''
            Acquire and set references from returned worker thread data.
            For details on the nature of return value, check compute_GLM.py 
            return description. 
        '''
        
        # Grab all GLM prediction and cell data
        self.cell_data = data[0]
        self.empty_cell = data[0][2]
        self.x = data[1]
        self.y = data[2]
        self.prediction = data[3]
        
        # Re-render list widget of available neurons
        if self.re_render:
            self.listWidget.clear()
            for i in range(self.empty_cell): 
                self.listWidget.addItem(QListWidgetItem(str(i+1)))
            self.re_render = False
            
        if self.graphType == 'Rate':
            self.rate_plot.axes.cla()
            self.rate_plot.axes.set_xlabel('Time')
            self.rate_plot.axes.set_ylabel('Rate (Hz)')
            self.rate_plot.axes.plot(self.x, self.y, linewidth=0.5)
            self.rate_plot.axes.plot(self.x, self.prediction, linewidth=1)
            self.rate_plot.draw()
        else:
            self.rate_plot.axes.cla()
            self.rate_plot.axes.set_xlabel('Speed')
            self.rate_plot.axes.set_ylabel('Rate (Hz)')
            self.rate_plot.axes.scatter(self.x, self.y, s=1)
            self.rate_plot.axes.plot(self.x, self.prediction, color='green', linewidth=1, )
            self.rate_plot.draw()
            
    # ------------------------------------------- # 
        
    def runWorkerThread(self):

        '''
            Launches worker thread to compute GLM prediction
        '''

        # Passes compute_all_map_data function to worker thread
        self.worker = Worker(compute_GLM, self.files, self.cell, self.ppm, self.family, self.graphType, cell_data=self.cell_data)
        # Connects PyQt5 signals to listeners
        self.worker.signals.return_data.connect(self.setData)
        self.worker.signals.progress.connect(self.progressBar)
        self.worker.signals.error.connect(self.errorOccured)
        # Starts the thread
        self.worker.start()

# =========================================================================== #

def main(): 
    
    '''
        Main function invokes application start.
    '''
    
    app = QApplication(sys.argv)
    screen = mainWindow()
    screen.show()
    sys.exit(app.exec_())
    
if __name__=="__main__":
    main()