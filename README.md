# CSE 599g1 - Deep Learning Final: Rice Image Classification

## Project Description

Rice is the most widely consumed staple food for over half of the world's population and is one of the most produced agricultural commodities worldwide. Rice consists of numerous genetic varieties. These varieties are separated from each other due to some of their features such a texture, shape, color. Using these visual features, it is possible to classify and evaluate the quality of rice grains. Using the Rice Image Dataset from Kaggle, we aim to explore various transfer learning approaches with neural network models to see how they generalize to classifying rice varieties. The specific neural network models that we apply transfer learning (using their PyTorch implementations) to are:

> AlexNet (a classic model)  
ResNet18 (a classic model)  
SqueezeNet (a more recent-ish model)  

We also use three different optimization strategies (SGD, Adadelta, RMSprop) for each model to see how they impact model performance for transfer learning.