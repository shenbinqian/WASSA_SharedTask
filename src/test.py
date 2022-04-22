# -*- coding: utf-8 -*-
import torch
from tester import write_tsv, test_tabular, test_DLrg, test_Embeddings, test_MTL, test
from preprocessing import combine_articles_to_essays, process_article
from utils import tokenization, weight_embeddings, get_test_data
import pickle
from score_tools import get_extracted_text
import numpy as np
# from RST_parse import extract_nuclei
# from embedding import extract_BERT_embeddings


def get_predictions(*input_data, method, problem, save_path):
    model = torch.load(save_path, map_location='cpu')
    assert len(input_data) == 1 or len(input_data) == 2
    if len(input_data) == 1:
        embeddings = input_data[0]
        pred = test_Embeddings(embeddings, model, problem)

    else:
        input_ids, input_mask = input_data
        if method == 'MTL':
            emp_label = torch.tensor(np.random.uniform(0, 7, 525), dtype=torch.long)
            distress_label = torch.tensor(np.random.uniform(0, 7, 525), dtype=torch.long)
            emo_label = torch.tensor(np.random.randint(7, size=525), dtype=torch.long)
            assert problem == 'rgNrg' or problem == 'rgNclf' or problem == 'clfNrg'
            if problem == 'rgNrg':
                targets = [emp_label, distress_label]
            else:
                targets = [emp_label, emo_label]

            pred = test_MTL(input_ids, input_mask, model, targets, problem=problem)

        elif method == 'tabular':
            pred = test_tabular(input_ids, input_mask, model, problem=problem)

        else:
            if problem == 'emotion' or problem == 'classification':
                pred = test(input_ids, input_mask, model, problem=problem)
            else:
                pred = test_DLrg(input_ids, input_mask, model)

    return pred


def main(analysis='None', method='finetune', problem='emotion', max_length=200,
         model_name='roberta-base', save_path='../models/saved_model.pt'):

    test_data = get_test_data()
    assert analysis == 'whole_article' or analysis == 'RST_parsing' or analysis == 'score_based' or analysis == 'None'
    assert method == 'finetune' or method == 'MTL' or method == 'tabular'

    if analysis == 'whole_article':
        articles = process_article()
        whole = combine_articles_to_essays(articles, test_data)

        input_ids, input_mask = tokenization(whole['essay'].values, MAX_LENGTH=max_length, model_name=model_name)
        pred = get_predictions(input_ids, input_mask, method=method, problem=problem, save_path=save_path)

    elif analysis == 'RST_parsing':
        '''
        nuclei = extract_nuclei(directory='../test_RST', check_alignment=True)
        whole_embeddings = extract_BERT_embeddings(test_data['essay'].values, MAX_LENGTH=200, 
                                                   model_name='roberta-base')
        nuclei_embeddings = extract_BERT_embeddings(nuclei, MAX_LENGTH=200, model_name='roberta-base')

        f = open('../data/test_embeddings.data', 'wb')
        pickle.dump(whole_embeddings, f)
        pickle.dump(nuclei_embeddings, f)
        f.close()
        '''
        of = open('../data/test_embeddings.data', 'rb')
        whole_embeddings = pickle.load(of)
        nuclei_embeddings = pickle.load(of)
        of.close()

        concat_embeddings = weight_embeddings(whole_embeddings, nuclei_embeddings, RST_rate=0.3)
        pred = get_predictions(concat_embeddings, method=method, problem=problem, save_path=save_path)

    elif analysis == 'score_based':
        filtered_articles_essays = get_extracted_text()
        test_data['essay'] = filtered_articles_essays
        input_ids, input_mask = tokenization(test_data['essay'].values, MAX_LENGTH=max_length, model_name=model_name)
        pred = get_predictions(input_ids, input_mask, method=method, problem=problem, save_path=save_path)

    else:
        input_ids, input_mask = tokenization(test_data['essay'].values, MAX_LENGTH=max_length, model_name=model_name)
        pred = get_predictions(input_ids, input_mask, method=method, problem=problem, save_path=save_path)

    return pred


if __name__ == '__main__':
    pred_emo = main(analysis='None', method='finetune', problem='emotion', max_length=200,
                    model_name='roberta-base', save_path='../models/Rb_emo_4_128.pt')
    write_tsv(pred_emo)

    pred_emp = main(analysis='None', method='finetune', problem='empathy', max_length=200, model_name='roberta-base',
                    save_path='../models/Rb_emp_4_128.pt')
    distress = main(analysis='None', method='finetune', problem='distress', max_length=200, model_name='roberta-base',
                    save_path='../models/Rb_dis_4_128.pt')
    pred_emp['distress'] = distress['preds'].values
    write_tsv(pred_emp)
