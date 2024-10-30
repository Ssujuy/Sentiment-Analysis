# Sentiment Analysis with Neural Network and GloVe Embeddings

This project leverages a neural network to perform sentiment classification on IMDB reviews. The workflow includes data preprocessing, neural network model training, evaluation, and performance visualization.

## Table of Contents
- [Project Setup and Preprocessing](#project-setup-and-preprocessing)
- [Code for GloVe Embedding and Data Tokenization](#code-for-glove-embedding-and-data-tokenization)
- [Building the Neural Network](#building-the-neural-network)
- [Training and Validation](#training-and-validation)
- [Result Analysis](#result-analysis)
- [Conclusion](#conclusion)

## Imports and  Prerequisites

- This section handles the required libraries and initial setup for running the sentiment analysis pipeline.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, classification_report, auc
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import io
import warnings
```
- Download NLTK Data: To tokenize text, the punkt dataset is downloaded to use NLTK’s tokenizer.

```python
nltk.download('punkt')
```

- File Paths: Define file paths for the dataset and the GloVe embedding file.

```python
dataPath = 'path_to_imdb_reviews.csv'
glove6btxtPath = 'path_to_glove.6B.100d.txt'
maxLength = 100
```

## Project Setup and Preprocessing

The first stage in our model pipeline is data preprocessing. This step prepares textual data for the neural network by embedding words using GloVe and processing review data into vectors.

```python
dataPath = 'path_to_imdb_reviews.csv'
glove6btxtPath = 'path_to_glove.6B.100d.txt'
maxLength = 100
dataset = pd.read_csv(dataPath, sep='\t')
```

- Loading and Embedding Words: IMDB reviews are loaded and split into training and validation sets. Word vectors from GloVe are then mapped for each word in the review dataset, enabling our neural network to learn relationships between words.

## Code for GloVe Embedding and Data Tokenization

```python
def fixSets(data, maxlen):
    # Iterates through sentences, creating embeddings for each word using GloVe
    # Outputs word vectors for each sentence to prepare data for training
```

## Building the Neural Network

- The neural network is a multi-layer perceptron (MLP) with ReLU6 activations. This configuration enables the model to generalize well on textual data.

```python
class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        # Define layers and activations
```

## Training and Validation

- The model trains over 200 epochs with the Mean Squared Error (MSE) loss function and Nadam optimizer, measuring both accuracy and loss for each epoch.

```python
for epoch in range(200):
    # Training loop with data augmentation and optimization steps
```

## Result Analysis

- Accuracy and Loss Curves: Plots generated during training visualize the model's accuracy and loss over epochs for both training and validation sets.
- Classification Report: Displays precision, recall, and F1 scores.
- ROC Curve: Shows model performance across various threshold levels, including micro- and macro-average curves.

## Conclusion

The final results, as seen in the classification report and ROC curves, show the effectiveness of the neural network in distinguishing between positive and negative sentiments in text data.

Through these analyses, we gain insights into areas of improvement for the model, such as potential tuning of learning rates or further refinement of text preprocessing methods. This notebook serves as a foundational approach to sentiment analysis with deep learning and GloVe embeddings, setting a pathway for enhanced text classification in similar applications.
