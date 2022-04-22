# WASSA Shared Task 2022

This is the project code for our participation in the WASSA Shared Task 2022 for empathy/distress prediction and emotion classification. Details of this shared task can be found at: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview. Method description can be found in our in-coming paper *SURREY-CTS-NLP at WASSA2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion*

## Data

The training, development and test datasets are contained in the directory './data', where GoEmotions datasets (Demszky et al., 2020) are also uploaded. RST parsing results (Zhang et al., 2021) of the training and test data are uploaded in './RST-output' and './test_RST respectively'.

## Models

Models can be saved and restored in './models', where 'save2.pt' is just an example.

## Results

Validation and test results can be saved in './ref' and './res'. Predictions of test data can be obtained by './src/test.py' and saved in './res' for uploading to CodaLab. During validation processing, ground truth targets can be saved in './ref' for evaluation by './src/evaluation.py'.

## Source Code

To train models, just run 'main.py'. There are several parameters that can be passed to select methods (fine-tuning or multi-task learning), problem (emotion classfication or empathy prediction) and data analysis techniques (RST parsing or sentiment/similarity score). Hyparameters for fine-tuneing a transformer model like learning rate or batch size can also be passed into the main function.

For training tabular models or using ensemble learning, 'tabular.py' and 'run_ensemble.py' can be used.


## Environments

Embeddings used in cosine similarity calculation are based on Universal Sentence Encoder (Cer et al., 2018) which requires Tensorflow environment. Other models are trained in Pytorch environment. Special attention needs to be paid on tabular.py, which requires python 3.7 and transformer 3.0, while other models were trained on python 3.9 and transformer 4.11.

## References

Daniel Cer, Yinfei Yang, Sheng yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2018. Universal sentence encoder. arXiv preprint.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. Goemotions: A dataset of fine-grained emotions. arXiv preprint.

Longyin Zhang, Fang Kong, and Guodong Zhou. 2021. Adversarial learning for discourse rhetorical structure parsing. pages 3946–3957. Association for Computational Linguistics.
