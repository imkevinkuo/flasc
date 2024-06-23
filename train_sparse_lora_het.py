import os
import argparse
from tqdm import tqdm
from copy import deepcopy

def str2bool(s):
    return s.lower() == 'true'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0',        type=str)
    parser.add_argument('--dir',        default='runs',     type=str)
    parser.add_argument('--name',       default='test',     type=str)
    parser.add_argument('--save',       default='false',     type=str)
    parser.add_argument('--dataset',    default='20newsgroups',  type=str)
    parser.add_argument('--iid-alpha',  default=0.1,  type=float)
    parser.add_argument('--clients',    default=350,       type=int)
    parser.add_argument('--model',      default='vit_b_16', type=str)
    parser.add_argument('--resume',     default=0,          type=int)
    parser.add_argument('--seed',       default=0,          type=int)
    parser.add_argument('--eval-freq',  default=20,         type=int)
    parser.add_argument('--eval-first',  default='false',      type=str)
    parser.add_argument('--eval-frac',  default=1,        type=float)
    parser.add_argument('--eval-masked',  default='true',      type=str)
    #
    parser.add_argument('--server-opt',       default='adam',  type=str)
    parser.add_argument('--server-lr',        default=5e-3,    type=float)
    parser.add_argument('--server-batch',     default=10,       type=int)
    parser.add_argument('--server-rounds',    default=200,      type=int)
    parser.add_argument('--client-lr',        default=5e-4,    type=float)
    parser.add_argument('--client-batch',     default=16,      type=int)
    parser.add_argument('--client-epochs',    default=1,       type=int)
    parser.add_argument('--client-freeze',    default='false',     type=str)
    parser.add_argument('--server-freeze',    default='false',     type=str)
    parser.add_argument('--syshet', default='rank', type=str) # 'rank' or 'density'
    parser.add_argument('--tiers',    default=3,       type=int)
    return parser.parse_args()

args = parse()
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['ROCR_VISIBLE_DEVICES'] = args.gpu
# os.environ['HIP_VISIBLE_DEVICES'] = args.gpu
if args.gpu != '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import random
import numpy as np
import torch
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
print(f"Visible GPUs: {torch.cuda.device_count()}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def get_topk_mask(x, density):
    mask = torch.zeros_like(x).bool()
    k = int(x.numel()*density)
    _, keep_idx = torch.topk(x, k=k)
    mask[keep_idx] = 1
    return mask

from train_utils import log_stats
def fl_train(save, run_dir, server_model, clients, valloader, testloader, test_batch,
             rounds, eval_freq, eval_first, eval_masked, server_opt,
             server_batch, server_lr, server_freeze, client_lr, client_epochs, client_freeze, 
             syshet, tiers):
    writer = tf.summary.create_file_writer(run_dir)
    pbar = tqdm(range(rounds))

    server_params = {n:p for n,p in server_model.named_parameters() if p.requires_grad}

    if server_opt == 'sgd':
        server_opt = torch.optim.SGD(server_params.values(), lr=server_lr)
    elif server_opt == 'adam':
        server_opt = torch.optim.AdamW(server_params.values(), lr=server_lr)
    else:
        raise ValueError()
    sched = torch.optim.lr_scheduler.StepLR(server_opt, step_size=1, gamma=1)

    eval_accu = 0
    def eval_loop(model, loader):        
        model.eval()
        stats_acc = {}
        for x,y in loader:
            with torch.no_grad():
                _, stats = test_batch(model, x, y)
            for k,v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0) + v
        stats_acc['loss'] /= stats_acc['count']
        return stats_acc
    
    if eval_first:
        log_stats(writer, "eval", stats, 0)
    
    cid_to_eff = [cid % tiers for cid in range(len(clients))]
    eff_to_cfg = [
        {'rank': 4**i, 'dl_density': 1/(4**(tiers-i-1)), 'ul_density': 1/(4**(tiers-i-1))} for i in range(tiers)
    ]
    # eff: 0,1,2 (1,4,16)
    
    for rnd in pbar:
        eff_to_mask = []
        for cfg in eff_to_cfg:
            cfg_mask = {}
            if syshet == 'density':
                server_params_flat = torch.cat([p.flatten() for n,p in server_params.items() if 'lora' in n])
                server_mask_flat = get_topk_mask(x=server_params_flat.abs(), density=cfg['dl_density'])
                curr = 0
                for n,p in server_params.items():
                    if 'lora' in n:
                        cfg_mask[n] = server_mask_flat[curr:curr+p.numel()].reshape(p.shape)
                        curr += p.numel()
                    else:
                        cfg_mask[n] = torch.ones_like(p)
            elif syshet == 'rank':
                for n,p in server_params.items():
                    p_mask = torch.ones_like(p)
                    if 'lora_A' in n:
                        p_mask[cfg['rank']:, :] = 0
                    elif 'lora_B' in n:
                        p_mask[:, cfg['rank']:] = 0
                    cfg_mask[n] = p_mask
            elif syshet == 'rankl1':
                for n,p in server_params.items():
                    p_mask = torch.ones_like(p.data)
                    if 'lora_A' in n:
                        _, zero_idx = torch.topk(p.data.norm(dim=1), k=4**(tiers-1) - cfg['rank'], largest=False)
                        p_mask[zero_idx, :] = 0
                    elif 'lora_B' in n:
                        _, zero_idx = torch.topk(p.data.norm(dim=0), k=4**(tiers-1) - cfg['rank'], largest=False)
                        p_mask[:, zero_idx] = 0
                    cfg_mask[n] = p_mask
            else:
                raise ValueError()
            eff_to_mask.append(cfg_mask)
        
        aggregate = None
        stats_acc = {}
        client_ids = torch.randperm(len(clients))[:server_batch]
        for i,client_id in enumerate(client_ids):
            eff = cid_to_eff[client_id]
            eff_cfg = eff_to_cfg[eff]
            dl_mask = eff_to_mask[eff]
            ul_density = eff_cfg['ul_density']
            # Download Sparsity
            client_model = deepcopy(server_model)
            for n,p in client_model.named_parameters():
                if p.requires_grad:
                    p.data = p.data*dl_mask[n]
            # Local Training
            client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
            client_loader = clients[client_id]
            client_acc = {}
            for epoch in range(client_epochs):
                for x,y in client_loader:
                    loss, stats = test_batch(client_model, x, y)
                    client_opt.zero_grad()
                    loss.backward()
                    if client_freeze:
                        for n,p in client_model.named_parameters():
                            if p.requires_grad:
                                p.grad *= dl_mask[n]
                    client_opt.step()
                    for k,v in stats.items():
                        client_acc[k] = client_acc.get(k, 0) + v
                    pbar.set_description(f"eval: {eval_accu} | client {i}, epoch {epoch} | loss {loss:.4f}")
            neg_client_delta = {n: (server_params[n].data*dl_mask[n]) - cp.data for n,cp 
                                in client_model.named_parameters() if cp.requires_grad}

            # Upload Sparsity
            if syshet == 'density':
                client_delta_flat = torch.cat([p.flatten() for p in neg_client_delta.values()])
                client_mask_flat = get_topk_mask(x=client_delta_flat.abs(), density=ul_density)
                curr = 0
                for n,p in neg_client_delta.items():
                    p *= client_mask_flat[curr:curr+p.numel()].reshape(p.shape)
                    curr += p.numel()
            # Aggregation
            if aggregate is None:
                aggregate = neg_client_delta
            else:
                for n, delta in neg_client_delta.items():
                    aggregate[n] += delta
            # Log last iteration
            for k,v in client_acc.items():
                stats_acc[k] = stats_acc.get(k, 0) + v
        # Server model update
        server_opt.zero_grad()
        for n, sp in server_params.items():
            sp.grad = aggregate[n] / server_batch
        server_opt.step()
        sched.step()
        # Eval and Logging
        if (rnd+1) % eval_freq == 0:
            for cfg, dl_mask in zip(eff_to_cfg, eff_to_mask):
                eval_model = deepcopy(server_model)
                for n,p in eval_model.named_parameters():
                    if p.requires_grad:
                        p.data = p.data*dl_mask[n]
                # if valloader is not None:
                #     log_stats(writer, "eval", eval_loop(eval_model, valloader), rnd+1)
                log_stats(writer, f"test_{cfg['dl_density']}", eval_loop(eval_model, testloader), rnd+1)
        
        stats_acc['loss'] /= stats_acc['count']
        log_stats(writer, "train", stats_acc, rnd+1)

        pbar.set_description(f"eval: {eval_accu}")
    # if save:
    #     torch.save({'delta': server_params, 'mask': server_mask}, f"{run_dir}/save.pt")

import data_utils
clients, valloader, testloader, test_batch = data_utils.build_dataset(
    args.dataset, args.client_batch, args.clients, args.iid_alpha, args.seed, args.eval_frac)

import models
model = models.build_model(args.dataset)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
models.add_adapters_dataset(args.dataset, model, lora_rank=4**(args.tiers-1), lora_alpha=16)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training {trainable} parameters ({100*trainable/total:.2f}% of original {total})")
model = model.cuda()

run_dir = f"{args.dir}/{args.name}"
os.makedirs(run_dir)
import json
with open(f"{run_dir}/args.json", 'w') as f:
  json.dump(vars(args), f, indent=4)
  print(f"Saved args to {run_dir}")

fl_train(str2bool(args.save),
    run_dir, model, clients, valloader, testloader, test_batch,
    rounds=args.server_rounds, 
    eval_freq=args.eval_freq,
    eval_first=str2bool(args.eval_first),
    eval_masked=str2bool(args.eval_masked),
    server_opt=args.server_opt,
    server_batch=args.server_batch, 
    server_lr=args.server_lr,
    server_freeze=str2bool(args.server_freeze),
    client_lr=args.client_lr, 
    client_epochs=args.client_epochs,
    client_freeze=str2bool(args.client_freeze),
    syshet=args.syshet,
    tiers=args.tiers,
)
