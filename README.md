# WASSA Shared Task 2022

This the project code for our participation in the WASSA Shared Task 2022 for empathy/distress prediction and emotion classification. Details of this shared task can be found at: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview. Method description can be found in our in-coming paper *SURREY-CTS-NLP at WASSA2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion*

## Data

The training, development and test datasets are contained in the directory './data', where GoEmotions datasets are also uploaded.

RST parsing results of the training and test data are uploaded in './RST-output' and './test_RST respectively'.

## Models

Models can be saved and restored in './models', where 'save2.pt' is just an example.

## Results

Validation and test results can be saved in './ref' and './res'. Predictions of test data can be obtained by './src/test.py' and saved in './res' for uploading to CodaLab. During validation processing, ground truth targets can be saved in './ref' for evaluation by './src/evaluation.py'.

## Source Code



## Environments

The default embedding method for statistical ML models is universal sentence encoder (USE) which requires Tensorflow environment, while deep learning methods require pytorch environment. Special attention needs to be paid on tabular.py, which requires python 3.7 and transformer 3.0.
