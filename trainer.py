from dataset import FewShotDataset 
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import copy
from scipy import stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaLearnerHelper():
  ''' Class to perform training and validation of the meta learner
       - model_learner (Learner): the Learner() model
       - model_meta (MetaLearner): the MetaLearner() model
       - k_shot (int): what k shot value to use
       - criterion (nn.loss): which loss function to use
     NOTE: majority of the code is my own code. I was aided in some areas by an implementation at 
     https://github.com/markdtw/meta-learning-lstm-pytorch/ to help structure the training functions.
     There are inherit similarities with this implementation and others since there are only so many
     ways to do the same thing.
  '''
  def __init__(self, model_learner, model_meta, k_shot, criterion=nn.NLLLoss(reduction='mean')):
    self.model_learner = model_learner.to(device)
    self.model_learner_outer = copy.deepcopy(self.model_learner)
    self.model_meta = model_meta.to(device)
    self.num_params = self.model_meta.num_params
    self.criterion = criterion
    self.k_shot = k_shot

    assert self.k_shot == 5 or self.k_shot == 1

    # create training datasets
    train_dataset = FewShotDataset(self.k_shot, 'train')
    val_dataset = FewShotDataset(self.k_shot, 'val')
    test_dataset = FewShotDataset(self.k_shot, 'test')

    self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    self.val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

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
  
  def _calc_conf_inter(self, all_acc):
    ''' Calculate the 95% confidence interval for the accuracy data
        all_acc (list): list of accuracies
        returns mean (float), size of the conf interval (float)
    '''
    sample_mean = np.mean(all_acc)
    sample_std = np.std(all_acc)
    sem = sample_std / np.sqrt(len(all_acc))
    ci = stats.t.interval(0.95, len(all_acc) - 1, loc=sample_mean, scale=sem)
    ci_size = (ci[1] - ci[0]) / 2 
    return sample_mean, ci_size

  def train_inner_learner(self, train_data):
    ''' Train the learner using updates from the meta learner
         - train_data (tuple): contains train images and labels
    '''
    hs_lstm =  None
    hs_meta_lstm = {'i': torch.zeros((self.num_params, self.model_meta.meta_lstm.hidden_size)).to(device),
                    'f': torch.zeros((self.num_params, self.model_meta.meta_lstm.hidden_size)).to(device),
                    'theta': self.model_meta.meta_lstm.theta}
    train_images, _, train_labels, _ = train_data

    train_images = train_images.squeeze().to(device)
    train_labels = train_labels.squeeze().to(device)
    training_losses = []

    # train learner for self.num_inner_epochs
    for epoch in range(self.num_inner_epochs): 
      # get new theta values from meta learner
      self.update_model_params(hs_meta_lstm['theta'])
      
      self.model_learner.zero_grad()

      out = self.model_learner(train_images)
      loss = self.criterion(out, train_labels)
      training_losses.append(loss.item())

      loss.backward()

      # get gradients of each param and use meta LSTM to update params
      flattened_grads = self.get_grads()
      hs_lstm, hs_meta_lstm = self.model_meta(flattened_grads, loss, hs_lstm, hs_meta_lstm)

    return hs_meta_lstm['theta']
  
  def train_meta_learner(self, num_epochs, model_path, grad_clip=0.25):
    ''' Train the entire meta learner.
        - num_epochs (int): number of epochs to train entire model for
        - grad_clip (float): gradient clipping value to use
    '''
    # use values from hyperparam tuning
    max_lr = 6.9e-4 if self.k_shot == 5 else 1.2e-3

    optimizer = torch.optim.Adam(self.model_meta.parameters(), lr=max_lr)

    # create cyclic learning rate with value found from tuning
    # step size is approx 5 times num iterations in one epoch
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=max_lr/4, 
                                                     max_lr=max_lr, step_size_up=2000,
                                                     mode='triangular2', cycle_momentum=False)

    running_acc, running_loss, running_count = 0.0, 0.0, 0
    all_loss = []
    all_val_loss = []
    loss_every = 100

    pbar = tqdm(range(num_epochs),
                desc="training", position=1)
    
    best_val_acc = 0
    
    idx = 0
    for epoch in pbar:
      # create a new dataset every iteration
      self.train_data_loader.dataset.shuffle_datasets()
      for train_data in self.train_data_loader:
        _, test_images, _, test_labels = train_data
        test_images = test_images.squeeze().to(device)
        test_labels = test_labels.squeeze().to(device)
        optimizer.zero_grad()
        self.erase_batch_stats()
        self.model_learner.train()
        self.model_learner_outer.train()

        theta = self.train_inner_learner(train_data)

        # transfer learned parameters to seperate learner model for loss calc
        self.model_learner_outer.transfer_params(self.model_learner, theta)
        test_output = self.model_learner_outer(test_images)
        loss = self.criterion(test_output, test_labels)

        pred_test_labels = torch.argmax(test_output, dim=1)
        correct = (pred_test_labels == test_labels).sum().item()
        acc = correct / test_labels.shape[0]

        running_loss += loss.item()
        running_acc += acc
        running_count += 1

        # calculate the meta learner gradients and update the grads accordingly
        meta_grads = torch.autograd.grad(loss, self.model_meta.parameters())
        for i, param in enumerate(self.model_meta.parameters()):
          param.grad = meta_grads[i]

        # Clip the gradients
        nn.utils.clip_grad_norm_(self.model_meta.parameters(), grad_clip)

        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f'''epoch: {epoch} loss: {running_loss/running_count:.3f} 
                               acc: {running_acc/running_count*100:.2f}''')
        
        if idx % loss_every == loss_every - 1:
          running_loss /= running_count
          all_loss.append(running_loss)
          print(f'epoch: {epoch:3d} iter: {idx} loss: {running_loss:6.4f} acc: {running_acc/running_count*100:5.2f}%')
          running_acc, running_loss, running_count = 0.0, 0.0, 0
        idx += 1
      
      ################################
      # validation after every epoch #
      ################################
      val_loss, val_acc = self.valid_meta_learner(is_valid=True, use_pbar=False)
      all_val_loss.append(val_loss)
      print(f'VALIDATION iter: {idx} loss: {val_loss:6.4f} acc: {val_acc[0]*100:5.2f}%')
      
      # save best model
      if val_acc[0] > best_val_acc:
        best_val_acc = val_acc[0]
        torch.save({
            'model_meta_state_dict': self.model_meta.state_dict(),
            'model_learner_state_dict': self.model_learner.state_dict(),
            'model_learner_outer_state_dict': self.model_learner_outer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': idx,
            'epoch': epoch,
            'acc': val_acc,
            'all_loss': all_loss,
            'valid_loss': all_val_loss
        }, f'{model_path}')
        
  def valid_meta_learner(self, is_valid, use_pbar=False):
    ''' Perform validation or testing on the meta learner
         - is_valid (bool): true if validation, false if testing set
         - use_pbar (bool): whether or not to display a progress bar
    '''
    name = 'valid' if is_valid else 'test'
    data_loader = self.val_data_loader if is_valid else self.test_data_loader

    running_acc, running_loss, running_count = 0.0, 0.0, 0
    all_acc = []
    loss_every = 100

    if use_pbar:
      pbar = tqdm(data_loader,
                  desc=name, position=1)
    else:
      pbar = data_loader

    # loop through all the images
    for idx, data in enumerate(pbar):
      _, test_images, _, test_labels = data
      test_images = test_images.squeeze().to(device)
      test_labels = test_labels.squeeze().to(device)

      self.erase_batch_stats()
      self.model_meta.eval()
      self.model_learner.train()

      # train inner learner and get theta values
      theta = self.train_inner_learner(data)
      
      # update the model params and get output
      self.update_model_params(theta)
      test_output = self.model_learner(test_images)
      loss = self.criterion(test_output, test_labels)

      # get predictions
      pred_test_labels = torch.argmax(test_output, dim=1)
      correct = (pred_test_labels == test_labels).sum().item()
      acc = correct / test_labels.shape[0]
      all_acc.append(acc)

      running_loss += loss.item()
      running_acc += acc
      running_count += 1

      if use_pbar:
        pbar.set_description(f'''{name} loss: {running_loss/running_count:.3f} 
                                acc: {running_acc/running_count*100:.2f}''')
    # get confidence interval
    conf_int = self._calc_conf_inter(all_acc) 
    
    return running_loss / running_count, conf_int
