import tqdm

import numpy as np

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel
from ._models import rmse, RMSELoss

class NeuralCollaborativeFiltering:

    def __init__(self, args, data=None, inference_model=False):
        super().__init__()

        self.criterion = RMSELoss()
        if data is not None:
            self.train_dataloader = data['train_dataloader']
            self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx=np.array((1, ), dtype=np.long)

        self.embed_dim = args.NCF_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.NCF_MLP_DIMS
        self.dropout = args.NCF_DROPOUT

        self.model = _NeuralCollaborativeFiltering(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        self.save_time = now_date + '_' + now_hour.replace(':', '')
        self.model_name = 'NCF_MODEL'
        self.min_val_loss = args.MIN_VAL_LOSS
        self.pretrained = args.PRETRAINED
        self.inference_model = inference_model


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained))
            print('--------------- PRETRAINED_MODEL LOAD ---------------')
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            
            if minimum_loss > rmse_score:
                minimum_loss = rmse_score
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{0}_{1}.pt'.format(self.save_time,self.model_name))
            print('epoch:', epoch, '| validation rmse:', rmse_score, '| minimum rmse:', minimum_loss)
        return minimum_loss


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        if self.min_val_loss:
            self.model.load_state_dict(torch.load('./models/{0}_{1}.pt'.format(self.save_time,self.model_name)))
            print('--------------- MIN_VAL_LOSS STATE LOAD ---------------')
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts

    def predict_once(self, data):
        self.model.eval()
        if self.inference_model:
            self.model.load_state_dict(torch.load(self.inference_model))
        with torch.no_grad():
                fields = data[0].to(self.device)
                y = self.model(fields)
        return y


class WideAndDeepModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.WDN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.WDN_MLP_DIMS
        self.dropout = args.WDN_DROPOUT

        self.model = _WideAndDeepModel(self.field_dims, self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        self.save_time = now_date + '_' + now_hour.replace(':', '')
        self.model_name = 'WDN_MODEL'
        self.min_val_loss = args.MIN_VAL_LOSS
        self.pretrained = args.PRETRAINED


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained))
            print('--------------- PRETRAINED_MODEL LOAD ---------------')
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            
            if minimum_loss > rmse_score:
                minimum_loss = rmse_score
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{0}_{1}.pt'.format(self.save_time,self.model_name))
            print('epoch:', epoch, '| validation rmse:', rmse_score, '| minimum rmse:', minimum_loss)
        return minimum_loss


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        if self.min_val_loss:
            self.model.load_state_dict(torch.load('./models/{0}_{1}.pt'.format(self.save_time,self.model_name)))
            print('--------------- MIN_VAL_LOSS STATE LOAD ---------------')
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts

    def predict_once(self, data):
        self.model.eval()
        if self.inference_model:
            self.model.load_state_dict(torch.load(self.inference_model))
        with torch.no_grad():
                fields = data[0].to(self.device)
                y = self.model(fields)
        return y 


class DeepCrossNetworkModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = _DeepCrossNetworkModel(self.field_dims, self.embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        self.save_time = now_date + '_' + now_hour.replace(':', '')
        self.model_name = 'DCN_MODEL'
        self.min_val_loss = args.MIN_VAL_LOSS
        self.pretrained = args.PRETRAINED


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained))
            print('--------------- PRETRAINED_MODEL LOAD ---------------')
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            
            if minimum_loss > rmse_score:
                minimum_loss = rmse_score
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{0}_{1}.pt'.format(self.save_time,self.model_name))
            print('epoch:', epoch, '| validation rmse:', rmse_score, '| minimum rmse:', minimum_loss)
        return minimum_loss


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        if self.min_val_loss:
            self.model.load_state_dict(torch.load('./models/{0}_{1}.pt'.format(self.save_time,self.model_name)))
            print('--------------- MIN_VAL_LOSS STATE LOAD ---------------')
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
