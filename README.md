## Prerequisites
The miniImageNet dataset can be downloaded from a link on my [google drive](https://drive.google.com/file/d/15scWjJZE31BBXSKEo8gpf9TZ2TYeyx_e/view?usp=sharing). Download this zip file and the extract it into the base directory of the repository into the folder `\miniImageNet`. Once this installed and the repository is cloned, then the model is ready to be trained or tested.

## Train the model
To train the model execute the following command
```bash
python3 main.py --mode=train --kshot=5 --epochs=60 --model=models/best_5_shot.pt
```
Where `kshot` is either 1 or 5 depending on if it is 1 shot or 5 shot, `epochs` is the number of epochs to train for, and `model` is the location of where the best model should be saved during training.

## Evaulate a trained model
To evaluate a trained model, execute the following command
```bash
python3 main.py --mode=test --kshot=5 --model=models/best_5_shot.pt
```

There exist two pretrained models in the `models` directory within this repository. These are the two models I trained for this project and represent the ones used to write my paper. To run the evaluation on these models run the following:
```bash
python3 main.py --mode=test --kshot=5 --model=models/best_5_shot_CYCLIC.pt
python3 main.py --mode=test --kshot=1 --model=models/best_1_shot_CYCLIC.pt
```
This script will output the testing accuracy as well as a plot of the loss from the training.

## Evaluate the baseline methods
To run and test the baseline methods that were used, run the following
```bash
python3 main.py --mode=baseline --kshot=5 --epochs=5 --model=empty
```
*Note*: you need to use `--model=empty` due to a quirk with the code. This does not actually save a model anywhere.

This will output the accuracy for each of the baeline methods.

## Run learning rate finetuning
To view the learning rate hyperparameter testing for the models, run the following
```bash
python3 main.py --mode=lrtuning --kshot=5 --model=empty
```
*Note*: you need to use `--model=empty` due to a quirk with the code. This does not actually save a model anywhere.

This will output a graph of the loss for the lr tuning for the given kshot as well as what the learning rate scheudler looked like throughout the training.