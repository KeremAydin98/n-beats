import numpy as np

# Configurations
WINDOW_SIZE = 7
HORIZON = 1

class Preprocess():

  def get_labelled_windows(self,x, horizon=HORIZON):

    return x[:,:-horizon], x[:,-horizon:]

  def get_full_windows(self, dataset, window_size = WINDOW_SIZE, horizon = HORIZON):

    """
    We will transform the dataset into windowed arrays by using numpy indexing
    """

    # Create one windows step
    window_steps = np.expand_dims(np.arange(window_size + horizon),axis=0)

    # Calculate the windows indexes in nested numpy arrays
    window_indexes = window_steps + np.expand_dims(np.arange(len(dataset) - (window_size + horizon -1)),axis=0).transpose()

    # Create the windowed array of the particular dataset by indexing
    windowed_array = dataset[window_indexes]

    # Slice it into window and target so that we can turn forecast problem into a supervised regression problem
    windows, labels = self.get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels