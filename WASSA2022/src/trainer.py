# -*- coding: UTF-8 -*-
import time
from utils import flat_accuracy, format_time
import torch
from scipy.stats.stats import pearsonr
import numpy as np
import torch.nn as nn


def backwards(loss, model, optimizer, warmup=None):
    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    optimizer.step()

    # Update the learning rate.
    if warmup:
        warmup.step()


def show_train_info(step, t0, train_dataloader, epoch_interval=40):
    # Progress update every 40 batches.
    if step % epoch_interval == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


class Trainer:
    def __init__(self, method='tabular', problem='regression'):
        self.method = method
        self.problem = problem

    def validate(self, dev_dataloader, model, device, statsRecord):
        print("")
        print("Running Validation...")
        statsRecord.write("Running Validation...\n")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_correlation = []
        correlation1 = []
        correlation2 = []
        total_eval_loss = 0

        if self.method == 'tabular':
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

                if logits.shape[1] != 1:
                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)

                else:
                    p_correlation = pearsonr(logits.squeeze(), label_ids.squeeze())
                    total_correlation.append(p_correlation[0])

            if self.problem == 'classification' or self.problem == 'emotion':
                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
                statsRecord.write("  Accuracy: {0:.2f}\n".format(avg_val_accuracy))
            else:
                print("Average Pearson Correlation: {0:.2f}".format(np.mean(total_correlation)))
                statsRecord.write("Average Pearson Correlation: {0:.2f}".format(np.mean(total_correlation)))

        elif self.method == 'MTL':
            for batch in dev_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels1 = batch[2].to(device)
                b_labels2 = batch[3].to(device)

                with torch.no_grad():
                    loss, outputs, _ = model(b_input_ids, b_input_mask, [b_labels1, b_labels2], problem=self.problem)

                # Accumulate the validation loss.
                total_eval_loss += loss

                # Move logits and labels to CPU
                outputs1 = outputs[0].detach().cpu().numpy()
                outputs2 = outputs[1].detach().cpu().numpy()
                label_ids1 = b_labels1.to('cpu').numpy()
                label_ids2 = b_labels2.to('cpu').numpy()

                assert self.problem == 'rgNrg' or self.problem == 'rgNclf' or self.problem == 'clfNrg'
                if self.problem == 'rgNrg':
                    p_correlation1 = pearsonr(outputs1.squeeze(), label_ids1.squeeze())
                    p_correlation2 = pearsonr(outputs2.squeeze(), label_ids2.squeeze())
                    correlation1.append(p_correlation1[0])
                    correlation2.append(p_correlation2[0])

                    print("Average Pearson Correlation for empathy: {0:.2f}".format(np.mean(correlation1)))
                    statsRecord.write("Average Pearson Correlation for empathy: {0:.2f}".format(np.mean(correlation1)))
                    print("Average Pearson Correlation for distress: {0:.2f}".format(np.mean(correlation2)))
                    statsRecord.write("Average Pearson Correlation for distress: {0:.2f}".format(np.mean(correlation2)))
                else:
                    p_correlation = pearsonr(outputs1.squeeze(), label_ids1.squeeze())
                    total_correlation.append(p_correlation[0])
                    total_eval_accuracy += flat_accuracy(outputs2, label_ids2)

                    avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
                    print("Average Pearson Correlation for empathy: {0:.2f}".format(np.mean(total_correlation)))
                    statsRecord.write("Average Pearson Correlation for empathy: {0:.2f}".format(
                        np.mean(total_correlation)))
                    print("Average Accuracy for Emotion Classification: {0:.2f}".format(avg_val_accuracy))
                    statsRecord.write("Average Accuracy for Emotion Classification: {0:.2f}".format(avg_val_accuracy))

        else:
            if self.problem == 'classification' or self.problem == 'emotion':
                for batch in dev_dataloader:
                    if len(batch) == 3:
                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)
                        with torch.no_grad():
                            result = model(input_ids=b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels,
                                           return_dict=True)

                    else:
                        b_embeddings = batch[0].to(device)
                        b_labels = batch[1].to(device)
                        with torch.no_grad():
                            result = model(inputs_embeds=b_embeddings, labels=b_labels, return_dict=True)

                    loss = result.loss
                    logits = result.logits

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

            else:
                for batch in dev_dataloader:
                    if len(batch) == 3:
                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)

                        with torch.no_grad():
                            outputs = model(input_ids=b_input_ids,
                                            attention_mask=b_input_mask)
                    else:
                        b_embeddings = batch[0].to(device)
                        b_labels = batch[1].to(device)

                        with torch.no_grad():
                            outputs = model(inputs_embeds=b_embeddings)

                    loss = nn.MSELoss()(outputs.squeeze(), b_labels.squeeze())

                    # Accumulate the validation loss and pearson correlation
                    total_eval_loss += loss.item()

                    outputs = outputs.detach().cpu().numpy()
                    score = b_labels.to('cpu').numpy()

                    p_correlation = pearsonr(outputs.squeeze(), score.squeeze())
                    total_correlation.append(p_correlation[0])

                print("Average Pearson Correlation: {0:.2f}".format(np.mean(total_correlation)))
                statsRecord.write("Average Pearson Correlation: {0:.2f}".format(np.mean(total_correlation)))

        # Calculate the average loss over all the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        statsRecord.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
        statsRecord.write("  Validation took: {:}\n".format(validation_time))

    def train(self, train_dataloader, dev_dataloader, model, epochs, optimizer, device, validate=True, warmup=None):
        # record total time for the train vali process
        total_t0 = time.time()

        # open a txt file to log records
        statsRecord = open('../stats.txt', 'a')

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

            if self.method == 'tabular':
                # For each batch of training data...
                for step, batch in enumerate(train_dataloader):

                    show_train_info(step, t0, train_dataloader)

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
                    backwards(loss, model, optimizer, warmup=warmup)

            elif self.method == 'MTL':
                # For each batch of training data...
                for step, batch in enumerate(train_dataloader):

                    show_train_info(step, t0, train_dataloader)

                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels1 = batch[2].to(device)
                    b_labels2 = batch[3].to(device)

                    model.zero_grad()

                    loss, _, _ = model(b_input_ids, b_input_mask, [b_labels1, b_labels2], problem=self.problem)

                    total_train_loss += loss
                    backwards(loss, model, optimizer, warmup=warmup)

            else:
                # For each batch of training data...
                for step, batch in enumerate(train_dataloader):

                    show_train_info(step, t0, train_dataloader)
                    model.zero_grad()
                    if len(batch) == 3:
                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)

                        if self.problem == 'classification' or self.problem == 'emotion':
                            result = model(input_ids=b_input_ids,
                                           attention_mask=b_input_mask,
                                           labels=b_labels,
                                           return_dict=True)
                            loss = result.loss
                        else:
                            outputs = model(input_ids=b_input_ids,
                                            attention_mask=b_input_mask)

                            loss = nn.MSELoss()(outputs.squeeze(), b_labels.squeeze())

                    else:
                        b_embeddings = batch[0].to(device)
                        b_labels = batch[1].to(device)

                        if self.problem == 'classification' or self.problem == 'emotion':
                            result = model(inputs_embeds=b_embeddings, labels=b_labels, return_dict=True)
                            loss = result.loss
                        else:
                            outputs = model(inputs_embeds=b_embeddings)
                            loss = nn.MSELoss()(outputs.squeeze(), b_labels.squeeze())

                    total_train_loss += loss.item()
                    backwards(loss, model, optimizer, warmup=warmup)

            # Calculate the average loss over all the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}".format(training_time))
            statsRecord.write("Average training loss: {0:.2f}\n".format(avg_train_loss))
            statsRecord.write("Training epoch took: {:}\n".format(training_time))

            if validate:
                self.validate(dev_dataloader, model, device, statsRecord)

        statsRecord.close()

        print("")
        print("Training complete!")
        print("Total training and validation took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
