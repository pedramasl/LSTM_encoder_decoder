
import torch
import pandas as pd
import numpy as np
from tkinter import Tk
import glob
import os
import warnings
from supersmoother import SuperSmoother

warnings.simplefilter("ignore")
shell = 'SS'
ID = 40
plateuTemp = 78 # top plateu temp of shell cure cycle
window_size = 20 # window size for moving average
startTemp = 40 # ramp up temp
endTemp = 50 # ramp down temp
maxRange = 2000 # max range of cure cycle
bigLeap = 300 # skip to get out of current cycle toward next cycle
maxTemp = 200
discardThresh = 100

def synthetic_data(Nt = 2000, tf = 80 * np.pi):

    folder_path =  "c:/Users/admin/Desktop/old/cure-thingy/WT20-PS/MM4"
    csv_files = glob.glob(folder_path + '/*'+shell+'*.csv')
    os.makedirs(folder_path + '/results', exist_ok=True)
    # for ID in range(1, 129):
    PV = 'PV' + str(ID)
    WSP = 'WSP' + str(ID)
    TSP = 'TSP' + str(ID) 

    pvCycles = [] 
    dates = []
    tgs = []
    tspCycles = []
    wspCycles = [] 
    pvMax = []

    for file_path in csv_files:
            try:
                date = file_path.split('/')[-1].split('_')[3].split(' ')[0]
                dates.append(date)
                df = pd.read_csv(file_path,delimiter=';')
                lastPV = df.columns[df.columns.str.startswith('PV')][-1].split('PV')[1]
                if ID > int(lastPV):  
                    continue
                df[PV] = df[PV][df[PV].astype(str).str.strip() != '']
                df[PV] = df[PV][df[PV].astype(float) != 0]
                df[PV] = df[PV][df[PV].astype(float) < maxTemp]
                df[TSP] = df[TSP][df[TSP].astype(str).str.strip() != '']
                df[TSP] = df[TSP][df[TSP].astype(float) < maxTemp]
                df[WSP] = df[WSP][df[WSP].astype(str).str.strip() != '']
                df[WSP] = df[WSP][df[WSP].astype(float) < maxTemp]
                temp = df[PV].dropna()
                PVdata = temp.reset_index(drop=True).to_numpy()
                temp = df[TSP].dropna()
                TSPdata = temp.reset_index(drop=True).to_numpy()
                temp = df[WSP].dropna()
                WSPdata = temp.reset_index(drop=True).to_numpy()
                
                startCure = [] 
                endCure = []
                PVelemlist = []
                TSPelemlist = []
                WSPelemlist = []
                rowindx = 0
                firstHit = False
                startIndx = 0
                endIndx = 0
                passed = False
                allSet = 0
                for _ in PVdata:
                    try:
                        if PVdata[rowindx]:
                            pass
                    except:
                        break
                    if (PVdata[rowindx] > plateuTemp) and not firstHit:
                        firstHit = True
                    if firstHit:
                        for i in range(rowindx,0,-1):
                            if (PVdata[i] < startTemp):
                                startIndx = i
                                allSet += 1
                                break
                        for i in range(rowindx,len(PVdata)):
                            if (PVdata[i] < endTemp):
                                endIndx = i
                                allSet += 1
                                break
                        PVelems = []
                        TSPelems = []
                        WSPelems = []
                        for a in range(startIndx,endIndx):
                            PVelems.append(PVdata[a])
                            TSPelems.append(TSPdata[a])
                            WSPelems.append(WSPdata[a])
                        if (abs(endIndx - startIndx) < maxRange) and (allSet == 2):
                            PVelemlist.append(PVelems)
                            TSPelemlist.append(TSPelems)
                            WSPelemlist.append(WSPelems)    
                            startCure.append(startIndx)
                            endCure.append(endIndx)
                        rowindx = endIndx + bigLeap
                        startIndx = 0
                        endIndx = 0
                        passed = False
                        firstHit = False
                    else:
                        rowindx += 1
                discard = []
                for j,p in enumerate(PVelemlist):
                    areaWSP = np.trapz(WSPelemlist[j])
                    areaTSP = np.trapz(TSPelemlist[j])
                    areaPV = np.trapz(PVelemlist[j])
                    if areaPV < areaWSP or areaPV < areaTSP:
                        discard.append(j)
                discard.sort(reverse=True)
                for a in discard:
                    startCure.pop(a)    
                    endCure.pop(a)  
            
                for cycle in range(len(startCure)):    
                    data = pd.Series(PVdata[startCure[cycle]:endCure[cycle]])
                    smoothed_data = SuperSmoother().fit(np.linspace(0,1,len(data)),data).predict(np.linspace(0,1,len(data)))
                    smoothed_data = [x for x in smoothed_data if x != '' and not np.isnan(x) and x != 0]

                    if len(smoothed_data) > 0:
                        pvCycles.append(smoothed_data)
                        tspCycles.append(TSPdata[startCure[cycle]:endCure[cycle]])
                        wspCycles.append(WSPdata[startCure[cycle]:endCure[cycle]]) 

                print(date + len(startCure)*' X')
                startCure = []
                endCure = []    
                PVelemlist = []
                df = None 
            except:
                continue 
    discard = []
    for cycle in pvCycles:
        pvMax.append(np.argmax(cycle))
    pvMaxMean = np.mean(pvMax)
    for a in range(len(pvCycles)):
        if  abs(pvMax[a] - pvMaxMean) > discardThresh:
            discard.append(a)
    discard.sort(reverse=True)
    for a in discard:
            pvCycles.pop(a)

    y = np.concatenate(pvCycles)
    y = pd.Series(y)
    y = np.array(y)
    Nt = len(y)
    tf = Nt
    t = np.linspace(0., tf, Nt)

    return t, y

def train_test_split(t, y, split = 0.8):

  '''
  
  split time series into train/test sets
  
  : param t:                      time array
  : para y:                       feature array
  : para split:                   percent of data to include in training set 
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''
  
  indx_split = int(split * len(y))
  indx_train = np.arange(0, indx_split)
  indx_test = np.arange(indx_split, len(y))
  
  t_train = t[indx_train]
  y_train = y[indx_train]
  y_train = y_train.reshape(-1, 1)
  
  t_test = t[indx_test]
  y_test = y[indx_test]
  y_test = y_test.reshape(-1, 1)
  
  return t_train, y_train, t_test, y_test 


def windowed_dataset(y, input_window = 5, output_window = 1, stride = 1, num_features = 1):
  
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stide:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])    
    
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + input_window
            end_y = start_y + output_window 
            Y[:, ii, ff] = y[start_y:end_y, ff]

    return X, Y


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):

    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 

    '''
    
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
    
    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
