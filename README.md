# AMLS assignment 23_24

This assignment consists of two parts; part A is a binary classification task on the Pneumonia dataset. Three classical machine learning methods are considered, with a final accuracy of 0.8494% using Random Forest. Part B is a multiclass classification task on the Pathmnist dataset. A 9-layered convolutional neural network (CNN) was deployed with 0.8189% accuracy.

## Requirement
The requirement packages are stored in 'requirements.txt', before starting to run scripts in this project, installs packages using the following code:

```bash
pip install -r requirements. txt
```

## Project Structure
This project contains for folders: A,B,Datasets and Dataset image.\\
A and B contains code modules for Task A and Task B, in particular, the 'dataset_X' files are used to generate images from the datasets and save thim in 'Dataset image'. The 'train_and_eval_X' files are used to complete the required task -- training models and evaluating them. The 'X_model' files are pretrain models that can be loaded from 'main.py'\\
Datasets contains the datasets used in the tasks.

## Running the scripts
To run task A and B, run 'main.py' in its root folder and follow the instruction. For each tasks, two options are provided: (1) Training a model from the beginning; (2) Loading the pretrained model and see the evaluation results.

