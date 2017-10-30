import tensorflow as tf
import numpy as np
import modelCLASS as md
import lstm_config as cf





def getData(dataDir):
 fileName_list = os.listdir(dataDir)
 dataX_List = []
 dataY_List = []
 fileNames_inds = collections.defaultdict(list)
 for fName in fileName_list:
   if fName.find("n2oMLData") == -1:
     continue
 
   ind = fName.find("n2oMLData")+10
   fileID = fName[ind:ind+15]
   fileNames_inds[float(fileID)].append(fName)
 
 for key,fNames in fileNames_inds.items():
   for fName in fNames:
     indA = fName.find("_ATOM")
     indB = fName.find("_BESSEL")
 
     #print(dataDir + fName)
     if indB != -1:
       #print(np.fromfile(dataDir + fName, dtype=np.float32)[:10])
       dataX_List.append(np.fromfile(dataDir + fName, dtype=np.float32))
     elif indA != -1:
       #print(np.fromfile(dataDir + fName, dtype=np.float32)[:10])
       dataY_List.append(np.fromfile(dataDir + fName, dtype=np.float32))
 
 #####  Finding size parameters from file names  #####
 NtimeSteps  = int(0)
 Nbessels    = int(0)
 Nradii      = int(0)
 Natoms      = int(0)
 Npos        = int(0)
 xSize       = int(0)
 ySize       = int(0)
 for fName in fileName_list:
   if fName.find("_BESSEL") != -1:
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
   if fName.find("_ATOM") != -1:
     ind1 = fName.find("Natoms")
     ind2 = fName.find("_Pos")
     Natoms = int(fName[ind1+7:ind2])
     ind1 = fName.find("_Job")
     Npos = int(fName[ind2+5:ind1])
     break
 
 xSize = (1 + Nbessels*Nradii)
 ySize = (Npos*Natoms)
 
 dataX = np.array(dataX_List)
 dataX[:,0] /= 100
 dataY = np.array(dataY_List)
 
 


class molMovLSTMCLASS(md.modelCLASS):
  
  def __init__(self, inp_config, inp_data, **inp_kwargs):
    md.modelCLASS.__init__(self, config = inp_config, data = inp_data, kwargs = inp_kwargs)

 
  def _initialize_placeHolders(self):
    self.X_placeHolder = tf.placeholder(tf.float32,
                          [None, self.config.time_length, self.config.X_length])
    self.Y_placeHolder = tf.placeholder(tf.float32,
                          [None, self.config.output_size])
    self.isTraining_placeHolder = tf.placeholder(tf.bool)
