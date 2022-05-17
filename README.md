# WASSA Shared Task 2022

This is the project code for our participation in the WASSA Shared Task 2022 for empathy/distress prediction and emotion classification. Details of this shared task can be found at: https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview. Method description can be found in our paper [*SURREY-CTS-NLP at WASSA2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion*](https://aclanthology.org/2022.wassa-1.29/)

## Citation

- Shenbin Qian, Constantin Orasan, Diptesh Kanojia, Hadeel Saadany, and Félix Do Carmo. 2022. SURREY-CTS-NLP at WASSA2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion. In Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis, pages 271–275, Dublin, Ireland. Association for Computational Linguistics.

```
@inproceedings{qian-etal-2022-surrey,
    title = "{SURREY}-{CTS}-{NLP} at {WASSA}2022: An Experiment of Discourse and Sentiment Analysis for the Prediction of Empathy, Distress and Emotion",
    author = "Qian, Shenbin  and
      Orasan, Constantin  and
      Kanojia, Diptesh  and
      Saadany, Hadeel  and
      Do Carmo, F{\'e}lix",
    booktitle = "Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment {\&} Social Media Analysis",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wassa-1.29",
    pages = "271--275",
    abstract = "This paper summarises the submissions our team, SURREY-CTS-NLP has made for the WASSA 2022 Shared Task for the prediction of empathy, distress and emotion. In this work, we tested different learning strategies, like ensemble learning and multi-task learning, as well as several large language models, but our primary focus was on analysing and extracting emotion-intensive features from both the essays in the training data and the news articles, to better predict empathy and distress scores from the perspective of discourse and sentiment analysis. We propose several text feature extraction schemes to compensate the small size of training examples for fine-tuning pretrained language models, including methods based on Rhetorical Structure Theory (RST) parsing, cosine similarity and sentiment score. Our best submissions achieve an average Pearson correlation score of 0.518 for the empathy prediction task and an F1 score of 0.571 for the emotion prediction task, indicating that using these schemes to extract emotion-intensive information can help improve model performance.",
}
```

## Data

The training, development and test datasets are contained in the directory './data', where filtered GoEmotions datasets (Demszky et al., 2020) are also uploaded. RST parsing results (Zhang et al., 2021) of the training and test data are uploaded in './RST-output' and './test_RST respectively'.

## Models

Models can be saved and restored in './models', where 'save2.pt' is just an example.

## Results

Validation and test results can be saved in './ref' and './res'. Predictions of test data can be obtained by './src/test.py' and saved in './res' for uploading to CodaLab. During the validation process, ground truth targets can be saved in './ref' for evaluation by './src/evaluation.py' provided by the organisers.

## Source Code

To train models, just run 'main.py'. There are several parameters that can be passed to select methods (fine-tuning or multi-task learning), problem (emotion classfication or empathy prediction) and data analysis techniques (RST parsing or sentiment/similarity score). Hyparameters for fine-tuneing a transformer model like learning rate or batch size can also be passed into the main function.

For training tabular models or using ensemble learning, 'tabular.py' and 'run_ensemble.py' can be used.


## Environments

Embeddings used in cosine similarity calculation are based on Universal Sentence Encoder (Cer et al., 2018) which requires Tensorflow environment. Other models are trained in Pytorch environment. Special attention needs to be paid on tabular.py, which requires python 3.7 and transformer 3.0, while other models were trained on python 3.9 and transformer 4.11.

## References

Daniel Cer, Yinfei Yang, Sheng yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2018. Universal sentence encoder. arXiv preprint.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. Goemotions: A dataset of fine-grained emotions. arXiv preprint.

Longyin Zhang, Fang Kong, and Guodong Zhou. 2021. Adversarial learning for discourse rhetorical structure parsing. pages 3946–3957. Association for Computational Linguistics.
