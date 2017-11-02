#import tensorflow as tf
import numpy as np
import molMovLSTM as mmLSTM
import lstm_config as cf
import random
import sys


if __name__ == '__main__':


  verbose = True

  ###  Configuration File  ###
  config = cf.configCLASS()

  ##################
  ###  Get Data  ###
  ##################

  print("Importing Data")

  ###  Import all the data  ####
  dataDir = "/reg/neh/home5/khegazy/analysis/machineLearning/simulation/output/"
  dataX, dataY = mmLSTM.getData(dataDir, config)

  if verbose:
    print("Original data X/Y shape:  {}  /  {}".format(dataX.shape, dataY.shape))

  ###  Split data  ###
  data = {}
  Nevents = dataX.shape[0]
  randInds = np.arange(Nevents)
  random.shuffle(randInds)
  ind1 = int(np.ceil(config.trainRatio*Nevents))
  ind2 = int(np.ceil((config.trainRatio + config.valRatio)*Nevents))
  ind3 = int(np.ceil((1 - config.testRatio)*Nevents))
  data["train_X"] = dataX[randInds[:ind1],:]
  data["train_Y"] = dataY[randInds[:ind1],:]
  data["val_X"]   = dataX[randInds[ind1:ind2],:]
  data["val_Y"]   = dataY[randInds[ind1:ind2],:]
  data["test_X"]  = dataX[randInds[ind2:],:]
  data["test_Y"]  = dataY[randInds[ind2],:]

  if verbose:
    print("Data was imported")
    print("X shape", data["train_X"].shape)
    print("Y shape", data["train_Y"].shape)


  ##############
  ###  LSTM  ###
  ##############

  print("Declare class")
  mmNet = mmLSTM.molMovLSTMCLASS(config, data)

  print("Building Graph")
  mmNet.build_graph()

  print("Training")
  mmNet.train()

  print("Plotting Results")
  mmNet.plot_results()

