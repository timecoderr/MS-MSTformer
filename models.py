import time

import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import prostformer as Prostformer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
import random
from sklearn.metrics import mean_absolute_error
from einops import rearrange, repeat
import shutil
import os

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class ctrNet(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        comfig = {

            'nyctaxi': {
                'dim': 32,
                'dim_1': 32 * 12,
                'exter_dim': 56,
                'num_frames': 4,
                'periods': 3,
                'image_size': (12, 16),
                'patch_size': (1, 1),
                'patch_size_1': (3, 4),
                'channels': 2,
                'depth': 6,
                'heads': 8,
                's': 16,
                'n': 12,
                'heads_1': 8,
                'dim_head': 4,
                'dim_head_1': 48,
                'attn_dropout': args.dropout,
                'ff_dropout': args.dropout,
                's1': 32,
                'n1': 6,
                'dim_2': 6 * 32,
                'dim_sd': 16 * 32,
            },

            'nycbike': {
                'dim': 32,
                'dim_1': 32 * 6,
                'exter_dim': 56,
                'num_frames': 4,
                'periods': 3,
                'image_size': (12, 16),
                'patch_size': (1, 1),
                'patch_size_1': (3, 2),
                'channels': 2,
                'depth': 6,
                'heads': 8,
                's': 32,
                'n': 6,
                'heads_1': 8,
                'dim_head': 4,
                'dim_head_1': 24,
                'attn_dropout': args.dropout,
                'ff_dropout': args.dropout,
                's1': 16,
                'n1': 12,
                'dim_2': 12 * 32,  # n1*d
                'dim_sd': 32 * 32,

            }

        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        logger.info(" device: %s, n_gpu: %s", device, args.n_gpu)
        param = comfig[args.index]
        logger.info("model config = %s", param)
        Model = MSMSTformer.MSMSTformer(
            dim=param['dim'],
            dim_1=param['dim_1'],
            exter_dim=param['exter_dim'],
            num_frames=param['num_frames'],
            periods=param['periods'],
            image_size=param['image_size'],
            patch_size=param['patch_size'],
            patch_size_1=param['patch_size_1'],
            channels=param['channels'],
            depth=param['depth'],
            heads=param['heads'],
            heads_1=param['heads_1'],
            dim_head=param['dim_head'],
            dim_head_1=param['dim_head_1'],
            attn_dropout=param['attn_dropout'],
            ff_dropout=param['ff_dropout'],
            s=param['s'],
            n=param['n'],
            s1=param['s1'],
            n1=param['n1'],
            dim_2=param['dim_2'],
            dim_sd=param['dim_sd'],
        )
        model = Model
        self.model = model.to(args.device)
        self.args = args
        self.logger = logger
        set_seed(args)

    def train(self, train_dataloader, dev_dataloader=None, best_loss=None):
        args = self.args
        logger = self.logger
        stop_epochs = args.early_stop_epoch
        early_stop_lis = np.zeros(stop_epochs)
        args.max_steps = args.epoch * len(train_dataloader)
        args.save_steps = len(train_dataloader) // 10
        args.warmup_steps = len(train_dataloader)
        args.logging_steps = len(train_dataloader)
        args.num_train_epochs = args.epoch

        optimizer = AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.08)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            len(train_dataloader) * args.num_train_epochs * 0.2), num_training_steps=int(
            len(train_dataloader) * args.num_train_epochs))

        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        model = self.model

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        if args.n_gpu != 0:
            logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size)
        logger.info("  Total optimization steps = %d", args.max_steps)

        global_step = 0
        tr_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0
        best_loss = float('inf') if best_loss == None else best_loss
        model.zero_grad()

        patience = 0
        for idx in range(args.num_train_epochs):
            tr_num = 0
            train_loss = 0
            stop_index = idx % stop_epochs

            if idx == 0:
                start_time = time.time()  
            for step, batch in enumerate(train_dataloader):

                video, external, y = [x.to(args.device) for x in batch]
                del batch
                model.train()
                loss, _ = model(video, external, y)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()

                if avg_loss == 0:
                    avg_loss = tr_loss
                avg_loss = round(train_loss / tr_num, 5)
                if (step + 1) % args.display_steps == 0:
                    logger.info("  epoch {} step {} loss {}".format(idx, step + 1, avg_loss))

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if idx == 0:
                end_time = time.time()  
                logger.info(
                        "Time used for the first epoch: {:.2f} seconds".format(end_time - start_time)) 

            if dev_dataloader is not None:
                model.eval()
                pres, y = self.infer(dev_dataloader)
                mse, mae = self.eval(pres, y)
                results = {}
                results['eval_loss'] = mse
                results['eval_mae'] = mae

                for key, value in results.items():
                    logger.info(" epoch %s, %s = %s", idx, key, round(float(value), 4))

                if results['eval_loss'] < best_loss:
                    early_stop_lis[stop_index] = 1
                    best_loss = results['eval_loss']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best loss : %s", round(float(best_loss), 4))
                    logger.info("  " + "*" * 20)
                    try:
                        os.system("mkdir -p {}".format(args.output_dir))
                    except:
                        pass
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                else:
                    early_stop_lis[stop_index] = -1
                if len(early_stop_lis) == -sum(early_stop_lis):
                    logger.info("eraly stop no changed (%s)", args.early_stop_epoch)
                    return
                else:
                    pass

    def infer(self, dev_dataloader):

        args = self.args
        model = self.model
        pre_lis = []
        y_lis = []
        model.eval()
        for batch in dev_dataloader:
            dev_video, dev_external, dev_y = [x.to(args.device) for x in batch]
            del batch
            with torch.no_grad():
                pre = model(dev_video, dev_external)
            pre_lis.append(pre.cpu())
            y_lis.append(dev_y.cpu())

        age_probs = torch.cat(pre_lis, 0)
        dev_y = torch.cat(y_lis, 0)
        return age_probs, dev_y

    def eval(self, preds, truth):
        def MAE(pred, gt):
            mae = torch.abs(pred - gt).mean()
            return mae

        loss = F.mse_loss(preds, truth)
        mae = MAE(preds, truth)
        return loss, mae

    def reload(self, i):  
        model = self.model
        args = self.args
        logger = self.logger


        args.load_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        logger.info("Load model from %s", args.load_model_path)

        new_dir = f"{args.index}_{args.time}_{args.time}"  
        dest_path = os.path.join(new_dir, f"pytorch{i}.bin")  

        os.makedirs(new_dir, exist_ok=True) 
        shutil.copy(args.load_model_path, dest_path)

        logger.info(f"Model copied to {dest_path}")  


        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(args.load_model_path))
