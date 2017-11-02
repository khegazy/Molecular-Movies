
class configCLASS():

  def __init__(self):

    # Training
    self.Nepochs = 10
    self.batch_size = 2
    self.sample_step = 5

    self.learning_rate = 1e-4
    self.decay_rate = 1
    self.minimizer = "Adam"

    # Data size
    self.Nframes    = 3
    self.Nbessels   = 3
    self.Nradii     = 20
    self.Nfeatures  = 10
    self.Natoms     = 3
    self.Nfeatures  = 1 + self.Nbessels*self.Nradii
    self.Noutputs   = 3*self.Natoms

    # Network size
    self.layer_size = 2
    self.Nhidden_states = 5
