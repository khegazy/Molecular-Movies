import tensorflow as tf
import numpy as np
import modelCLASS as md
import lstm_config as cf
import os
import collections


########################
###  Importing Data  ###
########################

def getData(dataDir, config):

  #####  Initialize Variables  #####
  fileName_list = os.listdir(dataDir)
  dataX_List = []
  dataY_List = []
  fileNames_inds = collections.defaultdict(list)

  #####  Finding size parameters from file names  #####
  NtimeSteps  = int(0)
  Nbessels    = int(0)
  Nradii      = int(0)
  NdiffBins   = int(0)
  Natoms      = int(0)
  Npos        = int(0)
  xSize       = int(0)
  ySize       = int(0)
  for fName in fileName_list:
    if (fName.find("_BESSEL") != -1) and (fName.find("Static") == -1):
      ind1 = fName.find("Ntime")
      ind2 = fName.find("_Nbessels")
      NtimeSteps = int(fName[ind1+6:ind2])
      ind1 = fName.find("_iso")
      Nbessels = int(fName[ind2+10:ind1])
      ind1 = fName.find("_Nradii")
      ind2 = fName.find("_Job")
      Nradii = int(fName[ind1+8:ind2])
      break

  for fName in fileName_list:
    if (fName.find("_ATOM") != -1) and (fName.find("Static") == -1):
      ind1 = fName.find("Natoms")
      ind2 = fName.find("_Pos")
      Natoms = int(fName[ind1+7:ind2])
      ind1 = fName.find("_NdiffBins")
      Npos = int(fName[ind2+5:ind1])
      ind2 = fName.find("_Job")
      NdiffBins = int(fName[ind1+11:ind2])
      break

  xSize = (1 + Nbessels*Nradii)
  ySize = Npos*Natoms + NdiffBins

  ### Check parameters from file match config
  if NtimeSteps != config.Nframes:
    raise RuntimeError("The number of time steps (frames) does not match!!!")
  if Nbessels != config.Nbessels:
    raise RuntimeError("The number of Bessel functions in data does not match!!!")
  if Natoms != config.Natoms:
    raise RuntimeError("The number of atoms in data does not match!!!")
  if xSize != config.Nfeatures:
    raise RuntimeError("The number of features in data does not match!!!")
  if ySize != config.Noutputs:
    raise RuntimeError("The number of outputs in data does not match!!!")


  #####  Retrieve Data From Files  #####

  ###  Make Dictionary of lists [XfileName, YfileName]  ###
  for fName in fileName_list:
    if fName.find("n2oMLData") == -1:
      continue

    if fName.find("Static") == -1:
      ind = fName.find("n2oMLData")+10
      fileID = fName[ind:ind+15]
      fileNames_inds[float(fileID)].append(fName)
    else:
      ind = "Static"
      fileNames_inds[ind].append(fName)

  ###  Get data from files  ###
  dataX_Static = None
  dataY_Static = None
  for key,fNames in fileNames_inds.items():
    if key != "Static":
      for fName in fNames:
        indA = fName.find("_ATOM")
        indB = fName.find("_BESSEL")

        if indB != -1:
          dataX_List.append(np.fromfile(dataDir + fName, dtype=np.float32))
        elif indA != -1:
          dataY_List.append(np.fromfile(dataDir + fName, dtype=np.float32))
    else:
      for fName in fNames:
        indA = fName.find("_ATOM")
        indB = fName.find("_BESSEL")

        if indB != -1:
          dataX_Static = np.fromfile(dataDir + fName, dtype=np.float32)
        elif indA != -1:
          dataY_Static = np.fromfile(dataDir + fName, dtype=np.float32)
 
  if (dataX_Static is None) or (dataY_Static is None):
    raise RuntimeError("Did not fill X and Y data for static pattern!!!")

  dataX = np.array(dataX_List)
  dataX = np.reshape(dataX, (dataX.shape[0], NtimeSteps, xSize))
  dataY = np.array(dataY_List)
  dataY = np.reshape(dataY, (dataY.shape[0], NtimeSteps, ySize))

  print("nan search", np.isnan(dataX).any(), np.isnan(dataY).any(), np.isnan(dataX_Static).any(), np.isnan(dataY_Static).any())

  #print("size: ",ySize)
  #print("original X")
  #print(dataX)
  #print("original Y")
  #print(dataY)
  #print("original static X")
  #print(dataX_Static)
  #print("original static Y")
  #print(dataY_Static)
  #print(dataX_Static, dataY_Static)
  return dataX, dataX_Static, dataY, dataY_Static 




##################################################
###  Defining Virtual Functions in modelCLASS  ###
##################################################


class molMovLSTMCLASS(md.modelCLASS):

  #####  Initialization  #####
  def __init__(self, inp_config, inp_data, **inp_kwargs):
    md.modelCLASS.__init__(self, config = inp_config, data = inp_data, kwargs = inp_kwargs)

 
  #####  Initialize Placeholders  #####
  def _initialize_placeHolders(self):
    self.X_placeHolder          = tf.placeholder(tf.float32,
                                      [None, self.config.Nframes, self.config.Nfeatures])
    self.Y_placeHolder          = tf.placeholder(tf.float32,
                                      [None, self.config.Nframes, self.config.Noutputs])
    self.X_static_placeHolder   = tf.placeholder(tf.float32, 
                                      [1, self.config.Nfeatures])
    self.Y_static_placeHolder   = tf.placeholder(tf.float32, 
                                      [1, self.config.Noutputs])
    self.isTraining_placeHolder = tf.placeholder(tf.bool)
    self.preProcess_placeHolder = tf.placeholder(tf.bool)


  #####  Prediction  #####
  """
  This is basically the full neural network up to the logits/predictions
  """
  def predict(self):

    if self.preProcess_placeHolder is True:

      #curInp = 
      ppFC1   = tf.layers.dense(inputs=self.X_placeHolder, units=self.Nbessels*int(3*self.Nradii/4))
      ppBN1   = tf.layers.batch_normalization(ppFC1, training=self.isTraining_placeHolder)
      ppRelu1 = tf.nn.relu(ppBN1)
      ppFC2   = tf.layers.dense(ppRelu1, units=self.Nbessels*int(self.Nradii/2))
      ppBN2   = tf.layers.batch_normalization(ppFC2, training=self.isTraining_placeHolder)
      ppRelu2 = tf.nn.relu(ppBN2)
      ppFC3   = tf.layers.dense(ppRelu2, units=self.Nbessels*int(self.Nradii/4))
      ppBN3   = tf.layers.batch_normalization(ppFC3, training=self.isTraining_placeHolder)
      ppRelu3 = tf.nn.relu(ppBN3)
      ppFC4   = tf.layers.dense(ppRelu3, units=self.Nbessels*int(self.Nradii/4))
      preProc = tf.layers.batch_normalization(ppFC4, training=self.isTraining_placeHolder)

      hidden_states, _ = tf.nn.dynamic_rnn(self.lstm_cells, preProc, dtype=tf.float32)

    else:
      hidden_states, _ = tf.nn.dynamic_rnn(self.lstm_cells, self.X_placeHolder, dtype=tf.float32)


    ###  Output NN  ###
    FC1   = tf.layers.dense(hidden_states, units=self.config.layer_size*5)
    BN1   = tf.layers.batch_normalization(FC1, training=self.isTraining_placeHolder)
    relu1 = tf.nn.relu(BN1)
    FC2   = tf.layers.dense(relu1, units=self.config.layer_size*2)
    BN2   = tf.layers.batch_normalization(FC2, training=self.isTraining_placeHolder)
    relu2 = tf.nn.relu(BN2)
    output= tf.layers.dense(relu2, self.config.Noutputs)

    return output


  #####  Total Loss Function  #####
  def calculate_loss(self, predictions, Y):
    if self.verbose:
      print("In Loss function")
    ###  Loss from comparing atom positions  ###
    positionLoss = tf.reduce_mean(tf.pow((predictions - Y), 2))

    ###  Loss from comparing diffraction diffraction pattern  ###

    if self.verbose:
      print("Exiting Loss function: {}".format(positionLoss))
    return positionLoss


  #####  Total Accuracy  #####
  def calculate_accuracy(self, predictions, Y):
    if self.verbose:
      print("In Accuracy function")
    return tf.reduce_mean(tf.abs((predictions - Y)/(Y + 0.0001)))


