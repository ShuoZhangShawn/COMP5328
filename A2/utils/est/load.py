import numpy as np
def load_data(file):
  """Loads the dataset."""



  dataset = np.load(file)
  X_train = dataset['X_tr']/255
  y_train = dataset['S_tr']
  X_test = dataset['X_ts']/255
  y_test = dataset['Y_ts']

  if X_train.ndim == 4:
    X_train = np.rollaxis(X_train, 3, 1)
    X_test = np.rollaxis(X_test, 3, 1)

  elif X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_train = np.rollaxis(X_train, 3, 1)
    X_test = np.rollaxis(X_test, 3, 1)

  return X_train, y_train, X_test, y_test