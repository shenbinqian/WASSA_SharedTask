# WASSA Shared Task 2022 for Empathy/Distress Prediction and Emotion Classification

This the project code for our participation in the WASSA Shared Task 2022 for empathy/distress prediction and emotion classification. Details of this shared task can be found at: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview. Method description can be found in our in-coming paper *SURREY-CTS-NLP at WASSA2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion*

## Data

CSV and TSV files contain all texts and other varibles for emotion classfication and empathy/distress prediction.

Data_viz.py can be used to visualise the distribution of these data. Preprocessing.py and utils.py can be used to wrangle these data for training purposes.

## Statistical Learning -- sML.py and ensemble.py

sML.py can be used for running of statistical machine learning models for emotion classification and empathy/distress prediction. Ensemble.py can be used for ensemble statistical ML models.

## Deep Learning -- DL.py and DLrg.py

DL.py and DLrg.py can be used for finetuning deep learning models (Roberta) for emotion classification and empathy/distress prediction.

## tabular.py

Tabular.py can be used for running experiments of combining essays, articles, numerical and categorical features for emotion classification and empathy/distress prediction.

## Environments

The default embedding method for statistical ML models is universal sentence encoder (USE) which requires Tensorflow environment, while deep learning methods require pytorch environment. Special attention needs to be paid on tabular.py, which requires python 3.7 and transformer 3.0.
