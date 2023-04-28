import random
from PIL import Image
import os
import glob
import torchvision.transforms as tvt
import torch
import json
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CreateDatasetStructure():
  ''' This class allows for the creation of the structure of the meta learning
      dataset and saves it to a json file.
       - label_file (string): location of the downloaded dataset labels file
  '''

  def __init__(self, label_file='miniImageNet/miniImageNetLabels.txt'):
    self.dataset_labels = {}

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            self.dataset_labels[line[0]] = line[2]

  def _create_dataset_structure(self, subset):
    ''' creates the dictionary structure of the zipped dataset
         - subset (string): either 'train', 'test', or 'val
    '''
    for id in glob.glob(f'miniImageNet/{subset}/*'):
      self.dataset_structure[subset][id] = glob.glob(f'{id}/*.jpg')
      self.lengths[subset].append(
          [len(self.dataset_structure[subset][id]), id])
      assert len(self.dataset_structure[subset][id]) == 600

      id = os.path.basename(id)
      assert id in self.dataset_labels
      assert id not in self.unique_classes
      self.unique_classes.add(id)

  def _get_splits(self, meta_dataset, sub_structure, sub_lengths, num_train_images,
                  num_test_images, num_classes):
    ''' This function creates the meta learning dataset splits for a given subset
        (train, test, val) of the original dataset. It randomizes each subdataset
        within the meta dataset to ensure randomness in the training data.
         - meta_dataset (dict): the structure to be saving the meta dataset
         - sub_structure (dict): subset of the original dataset structure
         - sub_lengths (list): length of each class within the subset
         - num_train_images (int): number of training images in each subdataset
         - num_test_images (int): number of test images in each subdataset
         - num_classes (int): number of classes in each subdataset
    '''
    idx = 0
    # sort the lengths to find the classes that still have the most images
    sub_lengths = sorted(sub_lengths, key=lambda x: (
        x[0], random.random()), reverse=True)
    while sub_lengths[num_classes-1][0] >= num_train_images + num_test_images:
      classes = [x[1] for x in sub_lengths[:num_classes]]
      meta_dataset[idx] = {'train': [], 'test': []}

      for j, curr_class in enumerate(classes):
        # get D_train images from class randomly with no replacement
        train_elements = random.sample(
            sub_structure[curr_class], num_train_images)
        for element in train_elements:
          meta_dataset[idx]['train'].append(
              {'img': element, 'label': os.path.basename(curr_class)})
          sub_structure[curr_class].remove(element)
          sub_lengths[j][0] -= 1

        # get D_test images from class randomly with no replacement
        test_elements = random.sample(
            sub_structure[curr_class], num_test_images)
        for element in test_elements:
          meta_dataset[idx]['test'].append(
              {'img': element, 'label': os.path.basename(curr_class)})
          sub_structure[curr_class].remove(element)
          sub_lengths[j][0] -= 1

        if len(sub_structure[curr_class]) == 0:
          del sub_structure[curr_class]

      sub_lengths = sorted(sub_lengths, key=lambda x: (
          x[0], random.random()), reverse=True)
      idx += 1

  def create_meta_learning_dataset(self, num_train_images, num_test_images=15, num_classes=5):
    ''' This function creates the entire meta learning dataset for a k-shot task
         - num_train_images (int): number of training images in each subdataset (k-shot)
         - num_test_images (int): number of test images in each subdataset
         - num_classes (int): number of classes in each subdataset
    '''
    meta_learning_dataset = {'train': {}, 'test': {}, 'val': {}}
    self.unique_classes = set()
    self.dataset_structure = {'train': {}, 'test': {}, 'val': {}}
    self.lengths = {'train': [], 'test': [], 'val': []}

    self._create_dataset_structure('train')
    self._create_dataset_structure('test')
    self._create_dataset_structure('val')

    self._get_splits(meta_learning_dataset['train'], self.dataset_structure['train'],
                     self.lengths['train'], num_train_images, num_test_images, num_classes)
    self._get_splits(meta_learning_dataset['test'], self.dataset_structure['test'],
                     self.lengths['test'], num_train_images, num_test_images, num_classes)
    self._get_splits(meta_learning_dataset['val'], self.dataset_structure['val'],
                     self.lengths['val'], num_train_images, num_test_images, num_classes)

    return meta_learning_dataset

  def save_to_json(self, meta_learning_dataset, filename):
    ''' Save the meta learning dataset to a json file for later use
         - meta_learning_dataset (dict): the structure of the dataset
         - filename (str): filename to store the json file
    '''
    with open(filename, 'w', encoding='utf-8') as f:
      json.dump(meta_learning_dataset, f, ensure_ascii=False, indent=4)


class FewShotDataset(torch.utils.data.Dataset):
  ''' Custom Torch Dataset to load in the images in the way described
      by the paper
       - k_shot (int): either 1 or 5, the number of images in each class
       - dataset_type (str): train, val, or test
       - image_size (int): size of the image
  '''

  def __init__(self, k_shot, dataset_type, image_size=84):
    super().__init__()
    self.k_shot = k_shot
    self.dataset_type = dataset_type
    self.image_size = image_size

    self.structure_creator = CreateDatasetStructure()

    # with open(f'meta_{k_shot}_shot_dataset.json', 'r') as f:
    # self.dataset_structure = json.load(f)[self.dataset_type]
    self.dataset_structure = self.structure_creator.create_meta_learning_dataset(
        self.k_shot)[self.dataset_type]

  def shuffle_datasets(self):
    '''Create a new grouping of datasets. This allows for a much larger number of unique datasets.
    '''
    self.dataset_structure = self.structure_creator.create_meta_learning_dataset(
        self.k_shot)[self.dataset_type]

  def __len__(self):
    ''' Get the length of the dataset
    '''
    return len(self.dataset_structure.keys())

  def _perform_transforms(self, image):
    ''' Perform the necessary transforms to get the images ready for the model.
        Different logic depending if we are training or not. The transforms were
        copied from the implementation of this paper located at:
        https://github.com/markdtw/meta-learning-lstm-pytorch/
         - image (PIL.Image): image to transform
        returns
         - transformed image (torch.Tensor)
    '''
    if self.dataset_type == 'train':
      transforms = tvt.Compose([
          tvt.Resize((self.image_size, self.image_size)),
          tvt.RandomResizedCrop(self.image_size),
          tvt.RandomHorizontalFlip(),
          tvt.ColorJitter(
              brightness=0.4,
              contrast=0.4,
              saturation=0.4,
              hue=0.2),
          tvt.ToTensor(),
          tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    else:
      transforms = tvt.Compose([
          tvt.Resize((self.image_size, self.image_size)),
          tvt.ToTensor(),
          tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    return transforms(image)

  def __getitem__(self, index):
    ''' Get the item in the dataset at the specified index
         - index (int): index to get item from
        Returns
         - train images (torch.Tensor)
         - test images (torch.Tensor)
         - train labels (torch.Tensor)
         - test labels (torch.Tensor)
    '''
    # load in the dataset info and shuffle the order of images
    # index = str(index)
    train_info = self.dataset_structure[index]['train']
    test_info = self.dataset_structure[index]['test']
    random.shuffle(train_info)
    random.shuffle(test_info)

    # create empty tensors to store the images and labels
    train_images = torch.zeros(
        (len(train_info), 3, self.image_size, self.image_size), dtype=torch.float)
    test_images = torch.zeros(
        (len(test_info), 3, self.image_size, self.image_size), dtype=torch.float)
    train_labels = torch.zeros((len(train_info)), dtype=torch.long)
    test_labels = torch.zeros((len(test_images)), dtype=torch.long)

    label_dict = {}
    max_idx = 0

    # save the images and labels for the training set
    for i, img in enumerate(train_info):
      # create label mappings
      label = img['label']
      if label not in label_dict:
        label_dict[label] = max_idx
        max_idx += 1

      # load in train images
      pil_img = Image.open(img['img'])
      if pil_img.mode != "RGB":
        pil_img = pil_img.convert(mode="RGB")

      transformed_img = self._perform_transforms(pil_img)
      train_images[i, :, :, :] = transformed_img
      train_labels[i] = label_dict[label]
    assert len(label_dict) == 5

    # save the images and labels for the test set
    for i, img in enumerate(test_info):
      # load in train images
      pil_img = Image.open(img['img'])
      if pil_img.mode != "RGB":
        pil_img = pil_img.convert(mode="RGB")

      transformed_img = self._perform_transforms(pil_img)
      test_images[i, :, :, :] = transformed_img
      test_labels[i] = label_dict[img['label']]

    return train_images, test_images, train_labels, test_labels
