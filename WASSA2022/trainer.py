# -*- coding: UTF-8 -*-
import time
import torch
from utils import format_time, flat_accuracy
from scipy.stats.stats import pearsonr
import numpy as np

def validate(dev_dataloader, model, device, statsRecord):
    print("")
    print("Running Validation...")
    statsRecord.write("Running Validation...\n")

    t0 = time.time()

    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in dev_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_input_numeric = batch['numerical_feats'].to(device)
        b_input_cat = batch['cat_feats'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            loss, logits, _ = model(input_ids=b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    numerical_feats=b_input_numeric,
                                    cat_feats=b_input_cat)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    statsRecord.write("  Accuracy: {0:.2f}\n".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(dev_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    statsRecord.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
    statsRecord.write("  Validation took: {:}\n".format(validation_time))

def validateRG(dev_dataloader, model, device, statsRecord):
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
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_input_numeric = batch['numerical_feats'].to(device)
        b_input_cat = batch['cat_feats'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            loss, logits, _ = model(input_ids=b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    numerical_feats=b_input_numeric,
                                    cat_feats=b_input_cat)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        p_correlation = pearsonr(logits.squeeze(), label_ids.squeeze())
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

def trainer(train_dataloader, dev_dataloader, model, epochs, optimizer, device, warmup=None):
    # record total time for the train vali process
    total_t0 = time.time()

    # open a txt file to log records
    statsRecord = open('stats.txt', 'a')

    # move model to device
    model.to(device)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        statsRecord.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # set model to be training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_input_numeric = batch['numerical_feats'].to(device)
            b_input_cat = batch['cat_feats'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()

            loss, _, _ = model(input_ids=b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               numerical_feats=b_input_numeric,
                               cat_feats=b_input_cat)

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            if warmup:
                warmup.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(training_time))
        statsRecord.write("Average training loss: {0:.2f}\n".format(avg_train_loss))
        statsRecord.write("Training epcoh took: {:}\n".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        validate(dev_dataloader, model, device, statsRecord)

    statsRecord.close()

    print("")
    print("Training complete!")
    print("Total training and validation took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))