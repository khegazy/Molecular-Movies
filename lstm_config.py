
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
    self.time_length = 3
    self.X_length = 10
    self.output_size = 3

    # Network size
    self.layer_size = 2
    self.Nhidden_states = 5
