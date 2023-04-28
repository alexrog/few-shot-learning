from models import Learner, MetaLearner
from dataset import FewShotDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogSpaceLR(torch.optim.lr_scheduler._LRScheduler):
  ''' Custom learning rate scheudler which outputs the learning rates
      in the log space between two values. Used to find the optimial learning
      rate per the cyclical searching method.
       - optimizer (torch.optim): optimizer that is used
       - start_lr (float): starting learning rate to use
       - end_lr (float): ending learning rate to use
       - num_steps (int): number of steps to take
       - last_epoch (int): last epoch to do the scheduler
  '''
  def __init__(self, optimizer, start_lr, end_lr, num_steps, last_epoch=-1):
    super(LogSpaceLR, self).__init__(optimizer, last_epoch)
    self.end_lr = end_lr
    assert start_lr < end_lr

    self.lrs = list(np.logspace(np.log10(start_lr),
                    np.log10(end_lr), num_steps))

  def get_lr(self):
    ''' Get the learning rate for all the optimizer param groups
    '''
    curr_step = self.last_epoch

    if curr_step >= len(self.lrs):
      return [self.end_lr for _ in self.optimizer.param_groups]

    curr_lr = self.lrs[curr_step]

    return [curr_lr for _ in self.optimizer.param_groups]


class LearningRateTuner():
  ''' Class to help tune the learning rate appropriately
       - k_shot (int): the number of training samples to use
       - criterion (nn.Loss): loss function to use
  '''
  def __init__(self, k_shot, criterion=nn.NLLLoss(reduction='mean')):
    # create the model learner and model meta learner
    self.model_learner = Learner()
    self.model_learner.to(device)

    theta_params = self.model_learner.get_params().unsqueeze(1)
    num_params = theta_params.shape[0]
    self.model_meta = MetaLearner(input_size=4, hidden_size=20,
                                  num_params=num_params, theta_params=theta_params)
    self.model_meta.to(device)

    self.model_learner_outer = copy.deepcopy(self.model_learner)
    self.num_params = self.model_meta.num_params
    self.criterion = criterion
    self.k_shot = k_shot
    assert self.k_shot == 5 or self.k_shot == 1

    # create the dataset to use
    train_dataset = FewShotDataset(self.k_shot, 'train')

    self.train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False)

    # run the tuning for 3 epochs
    self.num_outer_epochs = 3
    self.num_steps = len(self.train_data_loader) * self.num_outer_epochs
    self.optimizer = torch.optim.Adam(self.model_meta.parameters(), lr=1e-6)
    self.lr_scheduler = LogSpaceLR(
        self.optimizer, start_lr=1e-6, end_lr=1e-1, num_steps=self.num_steps)
    self.lr_order = []
    self.loss_order = []
    self.num_inner_epochs = 5 if self.k_shot == 5 else 12

  def get_grads(self):
    ''' Get the parameters for the model in one flattened tensor
    '''
    return torch.cat([p.grad.view(-1) for p in self.model_learner.parameters()])

  def erase_batch_stats(self):
    ''' Erase the batch statistics of the learner model. The paper suggests
        we do this on every new dataset D
    '''
    batch_norms = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    for module in self.model_learner.modules():
        if isinstance(module, tuple(batch_norms)):
            module.reset_running_stats()
    for module in self.model_learner_outer.modules():
        if isinstance(module, tuple(batch_norms)):
            module.reset_running_stats()

  def update_model_params(self, new_params):
    ''' Copies the params from new_params into the model params to update them
         - new_params (torch.Tensor): tensor of the flattened params for model
    '''
    i = 0
    for param in self.model_learner.parameters():
        j = i + param.numel()
        param.data.copy_(new_params[i:j].reshape(param.shape))
        i = j

  def train_inner_learner(self, train_data):
    ''' Train the inner learner CNN
         - train_data (tuple): the training images and labels
    '''
    hs_lstm = None
    hs_meta_lstm = {'i': torch.zeros((self.num_params, self.model_meta.meta_lstm.hidden_size)).to(device),
                    'f': torch.zeros((self.num_params, self.model_meta.meta_lstm.hidden_size)).to(device),
                    'theta': self.model_meta.meta_lstm.theta}
    train_images, _, train_labels, _ = train_data

    train_images = train_images.squeeze().to(device)
    train_labels = train_labels.squeeze().to(device)

    # loop through all the epochs
    for _ in range(self.num_inner_epochs):
      # update learner params using meta lstm
      self.update_model_params(hs_meta_lstm['theta'])
      self.model_learner.zero_grad()

      out = self.model_learner(train_images)
      loss = self.criterion(out, train_labels)

      loss.backward()

      # get grads and use meta learner to update params
      flattened_grads = self.get_grads()
      hs_lstm, hs_meta_lstm = self.model_meta(
          flattened_grads, loss, hs_lstm, hs_meta_lstm)

    return hs_meta_lstm['theta']

  def find_lr(self, grad_clip=0.25):
    ''' Function to run training to find the best learning rate
         - grad_clip (float): gradient clipping value to use
    '''
    running_acc, running_loss, running_count = 0.0, 0.0, 0
    self.lr_order = []
    self.loss_order = []

    pbar = tqdm(range(self.num_outer_epochs),
                desc="training", position=1)

    idx = 0

    for epoch in pbar:
      for train_data in self.train_data_loader:
        _, test_images, _, test_labels = train_data
        test_images = test_images.squeeze().to(device)
        test_labels = test_labels.squeeze().to(device)

        self.optimizer.zero_grad()
        self.erase_batch_stats()
        self.model_learner.train()
        self.model_learner_outer.train()

        # train inner learner and transfer params to another learner instance
        theta = self.train_inner_learner(train_data)
        self.model_learner_outer.transfer_params(self.model_learner, theta)
        test_output = self.model_learner_outer(test_images)
        loss = self.criterion(test_output, test_labels)

        # get predictions
        pred_test_labels = torch.argmax(test_output, dim=1)
        correct = (pred_test_labels == test_labels).sum().item()
        acc = correct / test_labels.shape[0]

        # get loss
        running_loss += loss.item()
        running_acc += acc
        running_count += 1

        # KEEP TRACK OF LEARNING RATE AND LOSS
        self.lr_order.append(self.lr_scheduler.get_lr())
        self.loss_order.append(loss.item())

        meta_grads = torch.autograd.grad(loss, self.model_meta.parameters())

        for i, param in enumerate(self.model_meta.parameters()):
          param.grad = meta_grads[i]
        # Clip the gradients
        nn.utils.clip_grad_norm_(self.model_meta.parameters(), grad_clip)

        self.optimizer.step()
        self.lr_scheduler.step()

        pbar.set_description(f'''epoch: {epoch} loss: {running_loss/running_count:.3f} 
                               acc: {running_acc/running_count*100:.2f}''')

        idx += 1

  def display_lrs(self):
    ''' Show graph of the learning rate tuning with the max and base lr
    '''
    # sliding window size
    N = 50

    # compute sliding window to make loss more smooth
    y = [np.mean(self.loss_order[i:i+N])
         for i in range(len(self.loss_order)-N)]

    # find the max and base lr
    max_lr = self.lr_order[np.argmin(y)][0]
    base_lr = max_lr / 4

    plt.axvline(x=max_lr, color='green', label=f'max_lr={max_lr:.1e}')
    plt.axvline(x=base_lr, color='red', label=f'base_lr={base_lr:.1e}')
    plt.legend()

    plt.loglog(self.lr_order[:-N], y)
    plt.title(f'Learning Rate Fine Tune for {self.k_shot} shot')
    plt.xlabel('Learning Rate')
    plt.ylabel('Meta Learner Loss')
    plt.show()

def view_lr_progression(k_shot, epochs):
  ''' Show a graph of the learning rate progression for the 1 shot and 5 shot training
  '''
  max_lr = 6.9e-4 if k_shot == 5 else 1.2e-3 # value found during my testing

  learner = Learner()
  optimizer = torch.optim.Adam(learner.parameters(), lr=max_lr)

  train_dataset = FewShotDataset(k_shot, 'train')
  train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

  # create the scheduler used
  lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=max_lr/4, 
                                                  max_lr=max_lr, step_size_up=2000,
                                                  mode='triangular2', cycle_momentum=False)
  lrs = []
  # create a fake training loop to show what the learning rate is doing
  for _ in range(epochs):
    for _ in range(len(train_data_loader)):
      lrs.append(lr_scheduler.get_last_lr())
      optimizer.step()
      lr_scheduler.step()

  # graph the progression
  fig, ax = plt.subplots()
  ax.plot(list(range(len(lrs))), lrs, linewidth=3, label=f'{k_shot}-shot')
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  ax.set_xlabel('Number of Iterations')
  ax.set_ylabel('Learning Rate')

  plt.legend()
  plt.show()