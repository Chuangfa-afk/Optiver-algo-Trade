# Optiver Realized Volatility Prediction

## Introduction
This repository contains our team's machine learning solution for the Optiver Realized Volatility Prediction Kaggle competition. Our approach ranked in the top 3%, achieving 166th place out of 4436 teams and earning a silver medal.

## Features
- Memory-efficient data handling
- Advanced feature engineering
- Time series normalization and sorting
- Implementation of various deep learning models:
  - Recurrent Neural Networks (RNN)
  - Long Short-Term Memory Networks (LSTM)
  - Gated Recurrent Units (GRU)
  - Transformer architecture
- Hyperparameter tuning for model optimization
- Model performance evaluation and visualization

## Installation
To set up the project environment:
pip install -r requirements.txt
It makes sure that you will have all the neccesary packages to run the `NN.py` script


## Usage
To replicate our analysis and model training:
1. Launch Jupyter Notebook or Jupyter Lab.
2. Open and execute the cells in `model_training.ipynb`.

## Data Preprocessing
The notebook details steps for reducing dataset memory usage, filling missing values, and normalizing features.

## Models and Training
We experimented with RNN, LSTM, GRU, and Transformer models, fine-tuning their architecture and hyperparameters for optimal results.

## Performance Visualization
The notebook includes visualizations comparing model performance metrics, specifically MAE, across various architectures.

## Results
The final section of the notebook provides insights into the competition's outcomes and the effectiveness of our predictive models.

## Collaboration
This project was a collaborative effort in compliance with the competition's data sharing and usage policies.

## Acknowledgements
We thank Optiver for the dataset and Kaggle for hosting the competition that challenged and inspired our work.

## Contact
For any queries regarding this project, please reach out through chuangfaliang2@gmail.com


## Deployment
Heroku serves as the primary platform for deploying and visualizing this model. Currently, due to an outage in the cloud service, Heroku is unavailable. As a result, all essential data and configurations required for the model's operation are contained within the NN.py file. This file includes detailed information on model parameters, setup instructions, and additional resources to ensure seamless integration and understanding of the model's functionality. This documentation is crucial for users looking to understand or utilize the model during the period when the Heroku platform is not operational.

This competition is in kaggle - Optiver Trading at the close
https://www.kaggle.com/competitions/optiver-trading-at-the-close



