# -*- coding: UTF-8 -*-

from utils import get_dataframe, get_train_dev, tokenization, data_loader
from utils import initialize_optimizer, scheduler
from utils import format_time
import time
from networks import RobertaRegressor
import torch
import numpy as np
import torch.nn as nn
from scipy.stats.stats import pearsonr

if __name__ == '__main__':
    df = get_dataframe()
    train_texts, dev_texts = get_train_dev(df['essay'])
    train_score, dev_score = get_train_dev(df['empathy'])
    train_score = torch.tensor(train_score.values, dtype=torch.double)
    dev_score = torch.tensor(dev_score.values, dtype=torch.double)

    train_ids, train_masks = tokenization(train_texts.values, model_name='roberta-large')
    dev_ids, dev_masks = tokenization(dev_texts.values, model_name='roberta-large')

    train_dataloader = data_loader(train_ids, train_masks, train_score, batch_size=2)
    dev_dataloader = data_loader(dev_ids, dev_masks, dev_score, batch_size=2)

    total_t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaRegressor()
    model.to(device)
    epochs = 4

    optimizer = initialize_optimizer(model)
    warmupSCH = scheduler(optimizer, epochs=epochs, dataloader=train_dataloader)
    loss_func = nn.MSELoss()

    # For each epoch...
    statsRecord = open('RGstats.txt', 'a')
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        statsRecord.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(input_ids=b_input_ids,
                           attention_masks=b_input_mask)
            loss = loss_func(outputs.squeeze(), b_labels.squeeze())

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            warmupSCH.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        statsRecord.write("  Average training loss: {0:.2f}".format(avg_train_loss))
        statsRecord.write("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")
        statsRecord.write("Running Validation...\n")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_corelation = []
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids=b_input_ids,
                               attention_masks=b_input_mask)

            loss = loss_func(outputs.squeeze(), b_labels.squeeze())

            # Accumulate the validation loss and pearson correlation
            total_eval_loss += loss.item()

            outputs = outputs.detach().cpu().numpy()
            score = b_labels.to('cpu').numpy()

            p_correlation = pearsonr(outputs.squeeze(), score.squeeze())
            total_corelation.append(p_correlation[0])

        print("Average Pearson Correlation: {0:.2f}".format(np.mean(total_corelation)))
        statsRecord.write("Average Pearson Correlation: {0:.2f}".format(np.mean(total_corelation)))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        statsRecord.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
        statsRecord.write("  Validation took: {:}\n".format(validation_time))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    statsRecord.close()