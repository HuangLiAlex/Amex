import os, gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR

from src.Transformer import TransformerModel
from src.early_stopper import early_stopper
from src.hyper_param import *
from src.metric import amex_metric
from src.GRU import GRU


def train_model_GRU():
    # SAVE TRUE AND OOF
    device = params['device']
    true = np.array([])
    oof = np.array([])

    for fold in range(5):   #apply 5 fold test

        # INDICES OF TRAIN AND VALID FOLDS
        valid_idx = [2 * fold + 1, 2 * fold + 2]
        train_idx = [x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if x not in valid_idx]

        print('#' * 25)
        print(f'### Fold {fold + 1} with valid files', valid_idx)

        # READ TRAIN DATA FROM DISK
        X_train = []
        y_train = []
        for k in train_idx:
            X_train.append(np.load(f'{PATH_TO_DATA}trans_data_{k}.npy'))
            y_train.append(pd.read_parquet(f'{PATH_TO_DATA}trans_targets_{k}.pqt'))
        X_train = np.concatenate(X_train, axis=0)
        X_train = np.delete(X_train, -1, axis=2)
        y_train = pd.concat(y_train).target.values
        print('### Training data shapes', X_train.shape, y_train.shape)

        # READ VALID DATA FROM DISK
        X_valid = []
        y_valid = []
        for k in valid_idx:
            X_valid.append(np.load(f'{PATH_TO_DATA}trans_data_{k}.npy'))
            y_valid.append(pd.read_parquet(f'{PATH_TO_DATA}trans_targets_{k}.pqt'))
        X_valid = np.concatenate(X_valid, axis=0)
        X_valid = np.delete(X_valid, -1, axis=2)
        y_valid = pd.concat(y_valid).target.values
        print('### Validation data shapes', X_valid.shape, y_valid.shape)
        print('#' * 25)

        # TRAIN MODEL
        # loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([118828, 340085])).float()).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        train_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(np.ones(X_train.shape[0]),
                                                                               num_samples=X_train.shape[0],
                                                                               replacement=False)
        train_dataloader = torch.utils.data.DataLoader(np.array(range(X_train.shape[0])),
                                                       batch_size=params['batch_size'], num_workers=0,
                                                       sampler=train_sample_strategy, drop_last=False)
        val_sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(np.ones(X_valid.shape[0]),
                                                                             num_samples=X_valid.shape[0],
                                                                             replacement=False)
        val_dataloader = torch.utils.data.DataLoader(np.array(range(X_valid.shape[0])), batch_size=params['batch_size'],
                                                     num_workers=0,
                                                     sampler=val_sample_strategy, drop_last=False)
        oof_predictions = torch.zeros(X_valid.shape[0], 2).float().to(device)
        model = eval(params['model'])(X_train.shape[-1]).to(device)
        lr = params['lr'] * np.sqrt(params['batch_size'] / 2048)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=params['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[3600, 5000, 6000], gamma=0.1)
        earlystoper = early_stopper(patience=params['early_stopping'], verbose=True)
        start_epoch = 0
        for epoch in range(start_epoch, params['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, input_seeds in enumerate(train_dataloader):
                input_seeds = input_seeds.type(torch.long)
                batch_inputs = torch.from_numpy(X_train[input_seeds]).to(device)
                batch_labels = torch.from_numpy(y_train[input_seeds]).to(device).long()
                model.hidden_state = model.init_hidden(len(input_seeds), device)
                train_batch_logits = model(batch_inputs)
                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                # tr_batch_pred = None

                if step % 50 == 0:
                    tr_batch_pred = torch.sum(
                        torch.argmax(train_batch_logits.clone().detach(), dim=1) == batch_labels) / batch_labels.shape[
                                        0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                    print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                          'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                       np.mean(train_loss_list),
                                                                                       average_precision_score(
                                                                                           batch_labels.cpu().numpy(),
                                                                                           score),
                                                                                       tr_batch_pred.detach(),
                                                                                       roc_auc_score(
                                                                                           batch_labels.cpu().numpy(),
                                                                                           score)))

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            # val_correct_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, input_seeds in enumerate(val_dataloader):
                    input_seeds = input_seeds.type(torch.long)
                    batch_inputs = torch.from_numpy(X_valid[input_seeds]).to(device)
                    batch_labels = torch.from_numpy(y_valid[input_seeds]).to(device).long()
                    model.hidden_state = model.init_hidden(len(input_seeds), device)
                    val_batch_logits = model(batch_inputs)
                    oof_predictions[input_seeds] = val_batch_logits
                    val_loss_list = val_loss_list + loss_fn(val_batch_logits, batch_labels)
                    val_batch_pred = torch.sum(torch.argmax(val_batch_logits, dim=1) == batch_labels) / torch.tensor(
                        batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    if step % 50 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                        print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                              'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                      step,
                                                                      val_loss_list / val_all_list,
                                                                      average_precision_score(
                                                                          batch_labels.cpu().numpy(), score),
                                                                      val_batch_pred.detach(),
                                                                      roc_auc_score(batch_labels.cpu().numpy(), score)))
                # tmp_predictions = model(test_feature).cpu().numpy()
            # infold_preds[fold] = tmp_predictions
            # test_predictions += tmp_predictions / params['n_fold']
            val_predictions = torch.softmax(oof_predictions.detach(), dim=-1)[:, 1].cpu().numpy()
            earlystoper.earlystop(val_loss_list, amex_metric(y_valid, val_predictions), model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_metric is: {:.7f}".format(earlystoper.best_cv))
        if not os.path.exists(PATH_TO_MODEL): os.makedirs(PATH_TO_MODEL)
        torch.save(earlystoper.best_model.to('cpu').state_dict(), f'{PATH_TO_MODEL}gru_fold_{fold + 1}.h5')

        # INFER VALID DATA
        print('Inferring validation data...')
        # mini-batch for validation
        val_loss_list = 0
        val_acc_list = 0
        # val_correct_list = 0
        val_all_list = 0
        model.load_state_dict(torch.load(f'{PATH_TO_MODEL}gru_fold_{fold + 1}.h5'))
        model.eval()
        with torch.no_grad():
            for step, input_seeds in enumerate(val_dataloader):
                input_seeds = input_seeds.type(torch.long)
                batch_inputs = torch.from_numpy(X_valid[input_seeds]).to(device)
                batch_labels = torch.from_numpy(y_valid[input_seeds]).to(device).long()
                model.hidden_state = model.init_hidden(len(input_seeds), device)
                val_batch_logits = model(batch_inputs)
                oof_predictions[input_seeds] = val_batch_logits
                val_loss_list = val_loss_list + loss_fn(val_batch_logits, batch_labels)
                val_batch_pred = torch.sum(torch.argmax(val_batch_logits, dim=1) == batch_labels) / torch.tensor(
                    batch_labels.shape[0])
                val_acc_list = val_acc_list + val_batch_pred * torch.tensor(batch_labels.shape[0])
                val_all_list = val_all_list + batch_labels.shape[0]
                if step % 50 == 0:
                    score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                    print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                          'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                  step,
                                                                  val_loss_list / val_all_list,
                                                                  average_precision_score(batch_labels.cpu().numpy(),
                                                                                          score),
                                                                  val_batch_pred.detach(),
                                                                  roc_auc_score(batch_labels.cpu().numpy(), score)))
        val_predictions = torch.softmax(oof_predictions.detach(), dim=-1)[:, 1].cpu().numpy()
        print()
        print(f'Fold {fold + 1} CV=', amex_metric(y_valid, val_predictions))
        print()
        true = np.concatenate([true, y_valid])
        oof = np.concatenate([oof, val_predictions])

        # CLEAN MEMORY
        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

    # PRINT OVERALL RESULTS
    print('#' * 25)
    print(f'Overall CV =', amex_metric(true, oof))


def train_model_transformer():
    device = params['device']
    DUMMY_TEXT = "test" # hl todo - to replace
    ntokens = len(DUMMY_TEXT.vocab.stoi) # the size of vocabulary
    emsize = 20 # embedding dimension
    nhid = 1 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 1 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    print(model)
    train_iter = 10

    for i in train_iter:
        print(f'Model input dimensions: {i.text.shape}')
        break
    print(f'Model output dimensions: {model(i.text).shape}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.train()
    for j in range(100):
        for i in train_iter:
            optimizer.zero_grad()
            output = model(i.text)
            loss = criterion(output.view(-1, output.shape[-1]), i.target.view(-1).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            print(f'Loss: {loss.item()}')
