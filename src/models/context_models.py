import tqdm

import numpy as np

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss


class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        self.save_time = now_date + '_' + now_hour.replace(':', '')
        self.model_name = 'FM_MODEL'
        self.min_val_loss = args.MIN_VAL_LOSS
        self.pretrained = args.PRETRAINED


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        val_total_loss = 0
        val_n = 0
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained))
            print('--------------- PRETRAINED_MODEL LOAD ---------------')
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()

            val_total_loss += rmse_score
            val_n += 1
            if minimum_loss > (val_total_loss/val_n):
                minimum_loss = (val_total_loss/val_n)
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


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        self.save_time = now_date + '_' + now_hour.replace(':', '')
        self.model_name = 'FFM_MODEL'
        self.min_val_loss = args.MIN_VAL_LOSS
        self.pretrained = args.PRETRAINED


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        val_total_loss = 0
        val_n = 0
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

            val_total_loss += rmse_score
            val_n += 1
            if minimum_loss > (val_total_loss/val_n):
                minimum_loss = (val_total_loss/val_n)
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
