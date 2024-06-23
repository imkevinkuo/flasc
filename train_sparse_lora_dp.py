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
    parser.add_argument('--dataset',    default='cifar10',  type=str)
    parser.add_argument('--iid-alpha',  default=0.1,  type=float)
    parser.add_argument('--clients',    default=500,       type=int)
    parser.add_argument('--model',      default='vit_b_16', type=str)
    parser.add_argument('--resume',     default=0,          type=int)
    parser.add_argument('--seed',       default=0,          type=int)
    parser.add_argument('--eval-freq',  default=10,         type=int)
    parser.add_argument('--eval-first',  default='false',      type=str)
    parser.add_argument('--eval-frac',  default=1,        type=float)
    parser.add_argument('--eval-masked',  default='true',      type=str)
    #
    parser.add_argument('--server-opt',       default='adam',  type=str)
    parser.add_argument('--server-lr',        default=1e-3,    type=float)
    parser.add_argument('--server-batch',     default=10,       type=int)
    parser.add_argument('--server-rounds',    default=200,      type=int)
    parser.add_argument('--client-lr',        default=1e-3,    type=float)
    parser.add_argument('--client-batch',     default=16,      type=int)
    parser.add_argument('--client-epochs',    default=1,       type=int)
    parser.add_argument('--client-freeze',    default='false',     type=str)
    parser.add_argument('--server-freeze',    default='false',     type=str)
    parser.add_argument('--freeze-a',         default='false',     type=str)
    parser.add_argument('--dl-density',       default=1.0,     type=float)
    parser.add_argument('--dl-density-decay', default=1.0,     type=float)
    parser.add_argument('--ul-density',       default=1.0,   type=float)
    parser.add_argument('--ul-density-decay', default=1.0,   type=float)
    parser.add_argument('--decay-freq', default=1,   type=int)
    parser.add_argument('--lora-rank',  default=16, type=int)
    parser.add_argument('--lora-alpha', default=16, type=int)
    parser.add_argument('--l2-clip-norm', default=0, type=float)
    parser.add_argument('--noise-multiplier', default=0, type=float)
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
             dl_density_init, dl_density_decay, ul_density_init, ul_density_decay, decay_freq,
             l2_clip_norm=0, noise_multiplier=0):
    writer = tf.summary.create_file_writer(run_dir)
    pbar = tqdm(range(rounds))

    server_params = {n:p for n,p in server_model.named_parameters() if p.requires_grad}
    server_mask = {n:torch.ones_like(p) for n,p in server_params.items()}
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
    
    for rnd in pbar:
        dl_density = dl_density_init*(dl_density_decay**(rnd//decay_freq))
        ul_density = ul_density_init*(ul_density_decay**(rnd//decay_freq))
        
        if (dl_density < 1 and not server_freeze) or (rnd == 1 and server_freeze): # one round of dense FT
            server_params_flat = torch.cat([p.flatten() for p in server_params.values()])
            server_mask_flat = get_topk_mask(x=server_params_flat.abs(), density=dl_density)
            curr = 0
            for n,m in server_mask.items():
                server_mask[n] = server_mask_flat[curr:curr+m.numel()].reshape(m.shape)
                curr += m.numel()
        
        aggregate = None
        stats_acc = {}
        client_ids = torch.randperm(len(clients))[:server_batch]
        for i,client_id in enumerate(client_ids):
            # Download Sparsity
            client_model = deepcopy(server_model)
            if dl_density < 1:
                for n,p in client_model.named_parameters():
                    if p.requires_grad:
                        p.data = p.data*server_mask[n]
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
                                p.grad *= server_mask[n]
                    client_opt.step()
                    for k,v in stats.items():
                        client_acc[k] = client_acc.get(k, 0) + v
                    pbar.set_description(f"eval: {eval_accu} | client {i}, epoch {epoch} | loss {loss:.4f}")
            if dl_density < 1:
                neg_client_delta = {n: (server_params[n].data*server_mask[n]) - cp.data for n,cp 
                                    in client_model.named_parameters() if cp.requires_grad}
            else:
                neg_client_delta = {n: server_params[n].data - cp.data for n,cp 
                                    in client_model.named_parameters() if cp.requires_grad}
            # Upload Sparsity
            if ul_density < 1:
                # why not log this?
                client_delta_flat = torch.cat([p.flatten() for p in neg_client_delta.values()])
                client_mask_flat = get_topk_mask(x=client_delta_flat.abs(), density=ul_density)
                curr = 0
                for n,p in neg_client_delta.items():
                    p *= client_mask_flat[curr:curr+p.numel()].reshape(p.shape)
                    curr += p.numel()
            # DP Clipping
            full_delta = torch.cat([delta.flatten() for delta in neg_client_delta.values()])
            delta_norm = torch.linalg.vector_norm(full_delta).item()
            if l2_clip_norm > 0:
                divisor = max(delta_norm / l2_clip_norm, 1.)
                for n,p in neg_client_delta.items():
                    p /= divisor
            # Aggregation
            if aggregate is None:
                aggregate = neg_client_delta
            else:
                for n, delta in neg_client_delta.items():
                    aggregate[n] += delta
            # Log last iteration
            client_acc['norm'] = delta_norm
            for k,v in client_acc.items():
                stats_acc[k] = stats_acc.get(k, 0) + v
        # DP Normalization and Noise
        if l2_clip_norm > 0:
            for n,p in aggregate.items():
                p /= l2_clip_norm
        if noise_multiplier > 0:
            for n,p in aggregate.items():
                p += noise_multiplier * torch.randn(*p.shape).cuda()
        # Server model update
        server_opt.zero_grad()
        for n, sp in server_params.items():
            sp.grad = aggregate[n] / server_batch
        server_opt.step()
        sched.step()
        # Eval and Logging
        if (rnd+1) % eval_freq == 0:
            eval_model = deepcopy(server_model)
            if eval_masked and dl_density < 1:
                for n,p in eval_model.named_parameters():
                    if p.requires_grad:
                        p.data *= server_mask[n]
            if valloader is not None:
                log_stats(writer, "eval", eval_loop(eval_model, valloader), rnd+1)
            log_stats(writer, "test", eval_loop(eval_model, testloader), rnd+1)
        
        with writer.as_default():
            tf.summary.scalar('density/download', dl_density, step=rnd+1)
            tf.summary.scalar('density/upload', ul_density, step=rnd+1)
        
        stats_acc['norm'] /= server_batch
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
models.add_adapters_dataset(args.dataset, model, args.lora_rank, args.lora_alpha)
if str2bool(args.freeze_a):
    for n,p in model.named_parameters():
        if "lora_A" in n:
            p.requires_grad = False
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
    dl_density_init=args.dl_density,
    dl_density_decay=args.dl_density_decay,
    ul_density_init=args.ul_density,
    ul_density_decay=args.ul_density_decay,
    decay_freq=args.decay_freq,
    l2_clip_norm=args.l2_clip_norm,
    noise_multiplier=args.noise_multiplier)
