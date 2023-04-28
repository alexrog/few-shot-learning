import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(nn.Module):
  ''' This is the inner learner CNN which the meta learner is trying to learn
      how to train. 
       - num_classes (int): number of classes for classifier
  '''

  def __init__(self, num_classes=5):
    super().__init__()

    self.conv_layers = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    )

    self.fc_layer = nn.Sequential(
        nn.Linear(32 * 5 * 5, num_classes),
        nn.LogSoftmax(dim=1)
    )

  def get_params(self):
    ''' Get the parameters of the model in a flattened tensor
    '''
    return torch.cat([p.view(-1) for p in self.parameters()], dim=0)

  def transfer_params(self, old_learner, theta):
    ''' Transfers the running mean and var from the old learner to the current one
        and copies the parameters from the flattened theta tensor into the model
        NOTE: This function is adapted from the implementation at:
        https://github.com/markdtw/meta-learning-lstm-pytorch/
        No matter what I tried, the only method that worked was the one presented
        in this implementation.
    '''
    self.load_state_dict(old_learner.state_dict())

    idx = 0
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        weight_len = m._parameters['weight'].view(-1).size(0)
        m._parameters['weight'] = theta[idx: idx +
                                        weight_len].view_as(m._parameters['weight']).clone()
        idx += weight_len
        if m._parameters['bias'] is not None:
          bias_len = m._parameters['bias'].view(-1).size(0)
          m._parameters['bias'] = theta[idx: idx +
                                        bias_len].view_as(m._parameters['bias']).clone()
          idx += bias_len

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(x.shape[0], -1)
    x = self.fc_layer(x)

    return x


class MetaLSTM(nn.Module):
  ''' The Meta LSTM cell as defined by the original paper.
       - input_size (int): size of input to gate layers
       - hidden_size (int): size of the hidden layers
       - num_params (int): number of params from the Learner
       - theta_params (torch.Tensor): current theta params for learner
  '''

  def __init__(self, input_size, hidden_size, num_params, theta_params):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_params = num_params

    in_size = input_size + 2
    out_size = hidden_size
    self.fc_wi = nn.Linear(in_size, out_size, bias=True)
    self.fc_wf = nn.Linear(in_size, out_size, bias=True)
    self.theta = nn.Parameter(torch.Tensor(num_params, hidden_size))
    self.sigmoid = nn.Sigmoid()

    self.initialize_weights(theta_params)

  def initialize_weights(self, theta_params):
    ''' The paper suggests intializing weight to certain values to ensure
        proper behavior at the beginning. The values chosen were the same as
        those in the official implementation located at:
        https://github.com/twitter-research/meta-learning-lstm/blob/master/model/lstm/meta-learner-lstm.lua
         - theta_params (torch.tensor): the theta params to initialize theta to be
    '''
    for p in self.parameters():
      nn.init.uniform_(p, -0.01, 0.01)

    self.fc_wi.bias.data.uniform_(-5, -4)
    self.fc_wf.bias.data.uniform_(4, 5)

    self.theta.data.copy_(theta_params)

  def forward(self, h_n_lstm, grads, hs_meta_lstm):
    i_0 = hs_meta_lstm['i']
    f_0 = hs_meta_lstm['f']
    theta_0 = hs_meta_lstm['theta']

    i_input = torch.cat((h_n_lstm, theta_0, i_0), dim=1)
    f_input = torch.cat((h_n_lstm, theta_0, f_0), dim=1)

    # i_t = sig(W_i @ [grad, loss, theta_t-1, i_t-1] + b_i)
    i_1 = self.sigmoid(self.fc_wi(i_input))

    # f_t = sig(W_f @ [grad, loss, theta_t-1, f_t-1] + b_f)
    f_1 = self.sigmoid(self.fc_wf(f_input))

    # theta_t = f_t * theta_t-1 - i_t * grad
    theta_1 = f_1 * theta_0 - i_1 * grads.unsqueeze(1)

    return {'i': i_1, 'f': f_1, 'theta': theta_1}


class MetaLearner(nn.Module):
  ''' The Meta-Learner class for learning how to train the Learner. It is an
      LSTM based approach as described by the original paper with a standard
      LSTM cell followed by a Meta LSTM cell which is a novel architecture.
       - input_size (int): size of input to LSTM cell
       - hidden_size (int): hidden size for LSTM cell
       - num_params (int): number of params in the learner
       - theta_params (torch.Tensor): all the current learner theta values
  '''

  def __init__(self, input_size, hidden_size, num_params, theta_params):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_params = num_params

    self.lstm = nn.LSTMCell(input_size, hidden_size)
    self.meta_lstm = MetaLSTM(hidden_size, 1, num_params, theta_params)

  def preprocessing(self, x, p=10):
    ''' Preprocess the x tensor as described in section 3.2 of paper
         - x (torch.Tensor): input tensor
         - p (int): parameter described by paper, they say to use 10
    '''
    eps = 1e-7  # ensure no torch.log(0) -> undefined
    preproc_tensor = torch.zeros((x.shape[0], 2)).to(device)
    true_loc = torch.abs(x) >= np.exp(-p)

    selected_sign = torch.sign(x[true_loc])
    selected_log = torch.log(torch.abs(x[true_loc]) + eps) / p
    preproc_tensor[true_loc.squeeze(0)] = torch.stack(
        (selected_log, selected_sign), dim=1)

    false_loc = ~true_loc
    neg_ones = (-torch.ones(x[false_loc].shape[0])).to(device)
    preproc_tensor[false_loc.squeeze(0)] = torch.stack(
        (neg_ones, np.exp(p) * x[false_loc]), dim=1)

    return preproc_tensor

  def forward(self, grads, loss, hs_lstm, hs_meta_lstm):
    loss_preproc = self.preprocessing(loss.unsqueeze(0).unsqueeze(0))
    grads_preproc = self.preprocessing(grads)

    lstm_input = torch.cat(
        (grads_preproc, loss_preproc.expand_as(grads_preproc)), dim=1)
    h_n_lstm, c_n_lstm = self.lstm(lstm_input, hs_lstm)

    hs_meta_lstm = self.meta_lstm(h_n_lstm, grads, hs_meta_lstm)

    return (h_n_lstm, c_n_lstm), hs_meta_lstm
