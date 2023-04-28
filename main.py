import argparse
from models import Learner, MetaLearner
from trainer import MetaLearnerHelper
from baseline import BaselineHelper
from lr_tuning import LearningRateTuner, view_lr_progression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.notebook import trange, tqdm


def train(num_epochs, k_shot, path):
  learner = Learner()
  theta_params = learner.get_params().unsqueeze(1)
  num_params = theta_params.shape[0]
  meta_learner = MetaLearner(input_size=4, hidden_size=20,
                             num_params=num_params, theta_params=theta_params)

  trainer_k_shot = MetaLearnerHelper(learner, meta_learner, k_shot)

  trainer_k_shot.train_meta_learner(num_epochs, path)


def plot_loss(all_loss, k_shot):
  plt.plot(list(range(1, len(all_loss)*100, 100)),
           all_loss, label=f'{k_shot} shot')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


def evaluate(k_shot, model_path):
  learner_curr = Learner()
  theta_params = learner_curr.get_params().unsqueeze(1)
  num_params = theta_params.shape[0]
  meta_learner_curr = MetaLearner(input_size=4, hidden_size=20,
                            num_params=num_params, theta_params=theta_params)

  best_model_dict = torch.load(model_path)
  learner_curr.load_state_dict(best_model_dict['model_learner_state_dict'])
  meta_learner_curr.load_state_dict(best_model_dict['model_meta_state_dict'])
  all_loss = best_model_dict['all_loss']
  valid_loss = best_model_dict['valid_loss']
  num_epoch = best_model_dict['epoch']

  trainer_valid = MetaLearnerHelper(learner_curr, meta_learner_curr, 5)
  loss, conf_int = trainer_valid.valid_meta_learner(is_valid=False, use_pbar=True)
  print(f'5 shot best\n  loss: {loss}\n  acc: {conf_int[0]*100:.2f}+/-{conf_int[1]*100:.2f}')

  plot_loss(all_loss, k_shot)

def lr_tuning(k_shot, epochs):
  lr_tuner = LearningRateTuner(k_shot)
  lr_tuner.find_lr()
  lr_tuner.display_lrs()

  view_lr_progression(k_shot, epochs)

def baseline(k_shot, epochs):
  baseline = BaselineHelper(k_shot=5, criterion=nn.CrossEntropyLoss())
  # get the CNN output
  baseline.train(num_epochs=epochs, lr=1e-3)

  # get the KNN output
  baseline.train_knn()

  # get the SVM output
  baseline.train_svm()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execute Code for Meta-Learner')

  parser.add_argument('--mode', type=str, choices=[
                      'train', 'test', 'lrtuning', 'baseline'], required=True, help='Mode to execute: train or test')
  parser.add_argument('--kshot', type=int, choices=[1,5], required=True,
                      help='Either 1 or 5 shot')
  parser.add_argument('--epochs', type=int, default=60,
                      help='Number of epochs to train for')
  parser.add_argument('--model', type=str, required=True, 
                      help='Model path to store if training, execute if evaluating')

  args = parser.parse_args()

  mode = args.mode
  k_shot = args.kshot
  epochs = args.epochs
  path = args.model

  if mode == 'train':
    train(epochs, k_shot, path)
  elif mode == 'test':
    evaluate(k_shot, path)
  elif mode == 'lrtuning':
    lr_tuning(k_shot, epochs)
  elif mode == 'baseline':
    baseline(k_shot, epochs)
