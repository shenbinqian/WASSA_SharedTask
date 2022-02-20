# WASSA_SharedTask

These are experiment codes for WASSA Shared Task 2022 for emotion classification and empathy/distress prediction. Details can be found at: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview.

## Data

CSV and TSV files contain all texts and other varibles for emotion classfication and empathy/distress prediction.

Data_viz.py can be used to visualise the distribution of these data. Preprocessing.py and utils.py can be used to wrangle these data for training purposes.

## Statistical Learning -- sML.py and ensemble.py

sML.py can be used for running of statistical machine learning models for emotion classification and empathy/distress prediction. Ensemble.py can be used for ensemble statistical ML models.

## Deep Learning -- DL.py and DLrg.py

DL.py and DLrg.py can be used for finetuning deep learning models (Roberta) for emotion classification and empathy/distress prediction.

## tabular.py

Tabular.py can be used for running experiments of combining essays, articles, numerical and categorical features for emotion classification and empathy/distress prediction.
