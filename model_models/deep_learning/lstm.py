import numpy as np

class LSTM:
  def __init__(self, input_dim, hidden_dim):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    
    # Initialize weights
    self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim)
    self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim)
    self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim)
    self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim)
    
    self.bf = np.random.randn(hidden_dim, 1)
    self.bi = np.random.randn(hidden_dim, 1)
    self.bc = np.random.randn(hidden_dim, 1)
    self.bo = np.random.randn(hidden_dim, 1)
    
  def forward(self, x, h_prev, c_prev):
    concat = np.concatenate((x,  h_prev), axis=0)
    
    # Input gate
    i = sigmoid(np.dot(self.Wi, concat) + self.bi)
    
    # Forget gate
    f = sigmoid(np.dot(self.Wf, concat) + self.bf)
    
    #Output gate
    o = sigmoid(np.dot(self.Wo, concat) + self.bo)
    
    #Cell state
    c_bar = np.tanh(np.dot(self.Wc, concat) + self.bc)
    c = f * c_prev + i * c_bar
    
    # Hidden state
    h = o * np.tanh(c)
    
    return h, c

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
