
class configCLASS():

  def __init__(self):

    # Training
    self.Nepochs = 100
    self.batch_size = 128
    self.sample_step = 3 
    self.preProcess = True

    self.learning_rate = 1e-4
    self.decay_rate = 1
    self.minimizer = "Adam"

    # Data size
    self.Nframes    = 11
    self.Nbessels   = 3
    self.Nradii     = 114
    self.NdiffBins  = 229
    self.Natoms     = 3
    self.Nfeatures  = 1 + self.Nbessels*self.Nradii
    self.Noutputs   = 3*self.Natoms + self.NdiffBins

    self.trainRatio = 0.6
    self.valRatio   = 0.2
    self.testRatio  = 0.2

    # Network size
    self.layer_size = 2
    self.Nhidden_states = 5
