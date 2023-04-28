from models import Learner 
from dataset import FewShotDataset
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineHelper():
  ''' Class to run the baseline cases
       - k_shot (int): which k shot to use
       - criterion (nn.Loss): loss function for the CNN
  '''
  def __init__(self, k_shot, criterion=nn.NLLLoss(reduction='mean')):
    self.criterion = criterion
    self.k_shot = k_shot
    assert self.k_shot == 5 or self.k_shot == 1

    # only use test dataset since we don't need to do any training beforehand
    test_dataset = FewShotDataset(self.k_shot, 'test')
    self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

  def _calc_conf_inter(self, all_acc):
    ''' Calculate the 95% confidence interval for the accuracy data
        all_acc (list): list of accuracies
        returns mean (float), size of the conf interval (float)
    '''
    sample_mean = np.mean(all_acc)
    sample_std = np.std(all_acc, ddof=1)
    sem = sample_std / np.sqrt(len(all_acc))
    ci = stats.t.interval(0.95, len(all_acc) - 1, loc=sample_mean, scale=sem)
    ci_size = (ci[1] - ci[0]) / 2 
    return sample_mean, ci_size

  def train_cnn(self, num_epochs, lr=1e-3):
    ''' Train the CNN baseline network
         - num_epochs (int): number of epochs to train for
         - lr (float): learning rate to use
    '''
    running_acc, running_loss, running_count = 0.0, 0.0, 0
    test_running_acc, test_running_loss, test_running_count = 0.0, 0.0, 0
    all_loss = []
    all_val_loss = []
    loss_every = 100

    pbar = tqdm(self.test_data_loader,
                desc="training", position=1)
    
    best_val_acc = 0
    
    idx = 0
    all_acc = []

    # create a new dataset every iteration
    for i, train_data in enumerate(pbar):
      train_images, test_images, train_labels, test_labels = train_data
      train_images = train_images.squeeze().to(device)
      train_labels = train_labels.squeeze().to(device)
      test_images = test_images.squeeze().to(device)
      test_labels = test_labels.squeeze().to(device)

      # need to create new learner for every dataset
      model = Learner().to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)

      for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.train()
        
        train_output = model(train_images)
        loss = self.criterion(train_output, train_labels)

        model.train()
        loss.backward()

        # get accuracy
        pred_train_labels = torch.argmax(train_output, dim=1)
        correct = (pred_train_labels == train_labels).sum().item()
        acc = correct / train_labels.shape[0]
        running_loss += loss.item()
        running_acc += acc
        running_count += 1

        optimizer.step()

      train_loss = running_loss / running_count

      # perform validation on the testing set
      model.eval()
      with torch.no_grad():
        test_output = model(test_images)
        loss = self.criterion(test_output, test_labels)

        pred_test_labels = torch.argmax(test_output, dim=1)
        correct = (pred_test_labels == test_labels).sum().item()
        acc = correct / test_labels.shape[0]
        all_acc.append(acc)

        test_running_loss += loss.item()
        test_running_acc += acc
        test_running_count += 1
        
      pbar.set_description(f'''epoch: {epoch} train_loss: {train_loss:.3f} test_loss: {test_running_loss/test_running_count:.3f} 
                              acc: {test_running_acc/test_running_count*100:.2f}''')
    
    acc, ci_size = self._calc_conf_inter(all_acc)
    print(f'CNN Accuracy: {acc*100:.2f}%+/- {ci_size*100:.2f}%')
    
  def train_knn(self):
    ''' Train a simple KNN to take in the input images and predict the classes
    '''

    total_correct = 0
    total = 0
    all_acc = []

    # loop through all the data 
    for i, (train_images, test_images, train_labels, test_labels) in enumerate(self.test_data_loader):
      # create a new kNN
      knn = KNeighborsClassifier(n_neighbors=5)
      train_images = train_images
      test_images = test_images

      # get the image embeddings 
      with torch.no_grad():
        train_embeddings = train_images.view(train_images.shape[1], -1)
        test_embeddings = test_images.view(test_images.shape[1], -1)
      # fit the data to the knn
      knn.fit(train_embeddings.numpy(), train_labels.squeeze().numpy())

      # predict the test set
      pred_labels = knn.predict(test_embeddings.numpy())

      # get accuracy
      total_correct = np.sum(pred_labels == test_labels.squeeze().numpy())
      total = pred_labels.shape[0]
      all_acc.append(total_correct/total)

    acc, ci_size = self._calc_conf_inter(all_acc)
    print(f'KNN Accuracy: {acc*100:.2f}%+/- {ci_size*100:.2f}%')

  def train_svm(self):
    ''' Train a support vector machine baseline classifier
    '''
    total_correct = 0
    total = 0
    all_acc = []

    # loop through all the training data
    for i, (train_images, test_images, train_labels, test_labels) in enumerate(self.test_data_loader):
      # create SVM
      svm = SVC()
      train_images = train_images
      test_images = test_images

      # get image embeddings
      with torch.no_grad():
        train_embeddings = train_images.view(train_images.shape[1], -1)
        test_embeddings = test_images.view(test_images.shape[1], -1)

      # fit the data to the svm
      svm.fit(train_embeddings.numpy(), train_labels.squeeze().numpy())

      # predict the test set
      pred_labels = svm.predict(test_embeddings.numpy())

      # get accuracy
      total_correct = np.sum(pred_labels == test_labels.squeeze().numpy())
      total = pred_labels.shape[0]
      all_acc.append(total_correct/total)
      
    acc, ci_size = self._calc_conf_inter(all_acc)
    print(f'SVM Accuracy: {acc*100:.2f}%+/- {ci_size*100:.2f}%')