import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class modelCLASS(object):
  """
  modelCLASS is a parent class to neural networks that encapsulates all 
  the logic necessary for training general nerual network archetectures.
  The graph is built by this class and the tensorflow session is held 
  by this class as well. Networks inhereting this class must inheret in
  the following way and define the following functions:
    ########################################
    #####  Build Neural Network Class  #####
    ########################################
    class NN(modelClass):
      
      def __init__(self, inp_config, inp_data, **inp_kwargs):
        modelCLASS.__init__(self, config = inp_config, data = inp_data, kwargs = inp_kwargs)
      def _initialize_placeHolders(self):
        Function to initialize place holders, only rand during __init__
      def predit(self):
        Function containing the network architecture to output the 
        predictions or logits
      def calculate_loss(self, preds, Y):
        Function to calculate loss given the predictions/logits from the 
        output of self.predict and the labels (Y)
      def calculate_accuracy(self, preds, Y):
        Function to calculate the accuracy for the predictions/logits 
        from self.predict against the labels(Y)
         
  Training a network using modelCLASS can be done after the previous 
  functions are defined. Once the class is initialized the graph must 
  first be built using self.build_graph(). After successfull graph 
  construction the network can be trained using self.train(). The 
  loss and accuracy history of the training can be plotted by calling 
  self.plot_results(). An example of training the network is given below.
  
      # Declare Class
      config = configCLASS()
      NN = neuralNetCLASS(inp_config=config)
      # Build Graph
      NN.build_graph()
      # Train
      NN.train()
      NN.plot_results()
  One can run the session to get the following values
    loss, acc, preds = NN.sess.run([NN.loss, NN.accuracy, NN.predictions],
                                   feed_dict = {
                                     self.X_placeHolder : data["type_X"],
                                     self.Y_placeHolder : data["type_Y"],
                                     self.isTraining_placeHolder : False})
  """

  def __init__(self, config, data, **kwargs):
    """
    Initialize
      config:
        A configure class that contains only variables used to set flags,
        data size, network parameters, parameterize the training, and 
        hold any other input variables into modelCLASS.
      data:
        Data to be trained and validated on. Data is a dictionary of of 
        [string : np.ndarray] pairs with the following keys:
          data["train_X"] = training sample features
          data["train_Y"] = training sample labels
          data["val_X"] = validation sample features
          data["val_Y"] = validation sample labels
      kwargs:
        keyword arguments for variables that are often changed, arguments 
        given this way will superceded arguments in the config file.
          NNtype: neural network type, given a type certain variables 
                  are mare accessible
          verbose: enable print statements during training
    """

    # Config
    self.config = config

    # Data
    self.data = data

    # Unpack keyword arguments
    self.verbose = kwargs.pop('verbose', True)

    # Flags
    self.hasTrained = False

    # Save History
    self.loss_history = {}
    self.accuracy_history = {}

    # Misc
    self.step = tf.Variable(0, trainable = False)
    self.Nbatches = int(np.ceil(data["train_X"].shape[0]/self.config.batch_size))


    #####  Graph Variables  #####

    self.graph_built = False

    self.predictions  = None
    self.loss         = None
    self.accuracy     = None
    self.train_step   = None
    self.sess         = None


    # Network cells
    self.lstm_cells = tf.contrib.rnn.MultiRNNCell(
                        [tf.contrib.rnn.LSTMCell(self.config.Nhidden_states) 
                          for _ in range(self.config.time_length)])

    # Placeholders
    self.X_placeHolder = None
    self.Y_placeHolder = None
    self.isTraining_placeHolder = None
    self._initialize_placeHolders()

    # Learning Rate
    self.learning_rate = -1
    if self.config.decay_rate != 1:
      self.initial_learning_rate = tf.train.exponential_decay(
                                    self.config.learning_rate,
                                    self.step,
                                    self.Nbatches,
                                    self.config.decay_rate)
    else:
      self.initial_learning_rate = self.config.learning_rate

    # Minimizer
    if self.config.minimizer == "Adam" :
      self.solver = tf.train.AdamOptimizer(
                                learning_rate = self.initial_learning_rate,
                                beta1         = 0.99,
                                beta2         = 0.9999)
    else:
      raise RuntimeError("Minimizer option " \
          + self.config.minimizer + " does not exist!")

    self._reset()


  #############################################################################
  def _initialize_placeHolders(self):
    """
    Initialize all place holders with size variables from self.config
    """
    pass


  #############################################################################
  def _reset(self):
    """
    Resets the graph before each time it is trained.
    """

    size = int(self.Nbatches*self.config.Nepochs/self.config.sample_step) 
    self.loss_history["train"] = np.zeros(size)
    self.loss_history["val"] = np.zeros(size)
    self.accuracy_history["train"] = np.zeros(size)
    self.accuracy_history["val"] = np.zeros(size)

    self.learning_rate = self.initial_learning_rate


  #############################################################################
  def optimize(self):
    """
    Training step called to update the training variables in order to minimize 
    self.loss with the minimization type of self.solver.
    """

    tstep = self.solver.minimize(self.loss, global_step = self.step) 

    # batch normalization in tensorflow requires this extra dependency
    # this is required to update the moving mean and moving variance variables
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
      tstep = self.solver.minimize(self.loss)

    return tstep


  #############################################################################
  def build_graph(self):
    """
    Builds the graph from the network specific functions. The graph is built 
    this to avoid calling the same part of the graph multiple times within 
    each self.sess.run call (this leads to errrors).
    """

    if self.graph_built:
      raise RuntimeError("Graph is already built, do not rebuild the graph!!!")
    else:
      self.graph_built = True

    self.predictions = self.predict()

    self.loss = self.calculate_loss(self.predictions, self.Y_placeHolder)

    self.accuracy = self.calculate_accuracy(self.predictions, self.Y_placeHolder)

    self.train_step = self.optimize()

    self.sess = tf.Session()


  #############################################################################
  def train(self):
    """
    Train the model by looping over epochs and making random batches for 
    each epoch. The loss and accuracy is saved in the history for each 
    self.config.sample_step minibatches.
    self.config.Nepoch: The number of epochs trained over
    self.config.batch_size: Maximimum size of each minibatch
    self.config.sample_step: Save and print (if verbose) the loss and 
        accuracy of each minibatch
    """  

    #####  Reset if needed  #####
    if self.hasTrained:
      self._reset()
    
    self.sess.run(tf.global_variables_initializer())

    train_size = self.data["train_X"].shape[0]
    indices = np.arange(train_size)
    count = 0

    ######  Loop over epochs  #####
    for epc in range(self.config.Nepochs):
      if self.verbose :
        print("\n\nEpoch: %i" % (epc))

      # shuffle indices for each epoch
      np.random.shuffle(indices)

      #####  Loop over mini batches  #####
      for ibt in range(self.Nbatches):
        minibatch_indices = indices[ibt*self.config.batch_size \
            : min((ibt + 1)*self.config.batch_size, train_size)]

        # Perform training step
        curLoss, curAcc, curPreds, _ = \
                  self.sess.run([self.loss, self.accuracy, self.predictions, self.train_step], 
                       feed_dict = { 
                         self.X_placeHolder : self.data["train_X"][minibatch_indices], 
                         self.Y_placeHolder : self.data["train_Y"][minibatch_indices], 
                         self.isTraining_placeHolder : True})

        # Sample training stats
        if count%self.config.sample_step == 0:
          ind = int(count/self.config.sample_step)
          self.loss_history["train"][ind] = curLoss
          self.accuracy_history["train"][ind] = curAcc

          curLoss, curAcc, curPreds = self.sess.run([self.loss, self.accuracy, self.predictions],
                                          feed_dict = { 
                                            self.X_placeHolder : self.data["val_X"], 
                                            self.Y_placeHolder : self.data["val_Y"], 
                                            self.isTraining_placeHolder : False})
          self.loss_history["val"][ind] = curLoss
          self.accuracy_history["val"][ind] = curAcc

          if self.verbose:
            print("Iteration(%i, %i)\tTrain[ Loss: %f\tAccuracy: %f]\tValidation[ Loss: %f\tAccuracy: %f]" % (epc, ibt, self.loss_history["train"][ind], self.accuracy_history["train"][ind], self.loss_history["val"][ind], self.accuracy_history["val"][ind]))

        count += 1


    loss_train, acc_train, preds_train = self.sess.run([self.loss, self.accuracy, self.predictions], 
                                         feed_dict = { 
                                           self.X_placeHolder : self.data["train_X"], 
                                           self.Y_placeHolder : self.data["train_Y"], 
                                           self.isTraining_placeHolder : False})

    loss_val, acc_val, preds_val       = self.sess.run([self.loss, self.accuracy, self.predictions], 
                                         feed_dict = { 
                                           self.X_placeHolder : self.data["val_X"], 
                                           self.Y_placeHolder : self.data["val_Y"], 
                                           self.isTraining_placeHolder : False})
    if self.verbose:
      print("\n\n")
      print("###########################")
      print("#####  Final Results  #####")
      print("###########################")
      print("\nTraining [ Loss: %f\t Accuracy: %f]" \
          % (loss_train, acc_train))
      print("Validation [ Loss: %f\t Accuracy: %f]" \
          % (loss_val, acc_val))
                              
    self.hasTrained = True
       

  #############################################################################
  def plot_results(self):
    """
    Plot the loss and accuracy history on both the train and validation
    datasets after training
    """

    f1, (ax1) = plt.subplots()
    h1, = ax1.plot(self.loss_history["train"], "b-", label = "Loss - Train")
    h2, = ax1.plot(self.loss_history["val"], "b.", label = "Loss - Validation")

    ax1.set_ylabel("Loss", color = "b")
    ax1.tick_params("y", color = "b")
    ax1.set_xlabel("Training Steps [{}]".format(self.config.sample_step))

    ax2 = ax1.twinx()
    h3, = ax2.plot(self.accuracy_history["train"], "r-", \
        label = "Accuracy - Train")
    h4, = ax2.plot(self.accuracy_history["val"], "r.", \
        label = "Accuracy - Validation")

    ax2.set_ylabel("Accuracy", color = "r")
    ax2.tick_params("y", color = "r")

    #plt.legend([h1, h2, h3, h4])
    f1.tight_layout()
    plt.savefig("trainingHistory.png")

    plt.show()


  #####################################################
  #####  Neural Network Specific Graph Functions  #####
  #####################################################

  def predict(self):
    """
    This function contains the neural network architecture and runs on the 
    data fed into the feed_dict for self.X_placeHolder and self.Y_placeHolder.
    The output is the prediction/logits of the network that will be fed into 
    the loss and accuracy functions.
    """
    pass 


  def calculate_loss(self, preds, Y):
    """
    Calculates the loss for the given predictions/logits (preds) and labels
    (Y). This function with input given by the X and Y placeholders is 
    given the name self.loss:
      
      self.loss = self.calculate_loss(self.predict(), self.Y_placeHolder)
    """
    pass


  def calculate_accuracy(self, preds, Y):
    """
    Calculates the accuracy for the given predictions/logits (preds) and labels
    (Y). This function with input given by the X and Y placeholders is 
    given the name self.accuracy:
      
      self.accuracy = self.calculate_accuracy(self.predict(), self.Y_placeHolder)
    """
    pass
