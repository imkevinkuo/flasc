import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
import os
import json
import numpy as np
from scipy import stats
from PIL import Image
from train_utils import test_batch_cls, test_batch_nwp

DATA = "."

def test_batch_ds(model, x, y, dataset):
    if dataset == "reddit":
        loss, stats = test_batch_nwp(model, x.cuda())
    elif dataset == "flair":
        loss, stats = test_batch_cls(model, x.cuda(), y.cuda(), multilabel=True)
    elif dataset == "cifar10" or dataset == "20newsgroups":
        loss, stats = test_batch_cls(model, x.cuda(), y.cuda(), multilabel=False)
    return loss, stats


def build_dataset(dataset, batch_size, n_clients, alpha=-1, seed=0, eval_frac=1):
    valset = None
    if dataset == 'cifar10':
        clients, valset, testset = build_cifar10(n_clients, alpha, seed)
        TEST_BATCH = 32
    elif dataset == '20newsgroups':
        clients, valset, testset = build_20newsgroups(n_clients, alpha, seed)
        TEST_BATCH = 16
    elif dataset == 'reddit':
        clients, valset, testset = build_reddit(alpha, seed)
        TEST_BATCH = 16
    elif dataset == 'flair':
        clients, valset, testset = build_flair(eval_frac)
        # clientloaders = clients
        TEST_BATCH = 16
    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True, num_workers=0) for client in clients]
    if valset is not None:
        valloader = DataLoader(valset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    else:
        valloader = None
    testloader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=1)
    def test_batch(model, x, y):
        return test_batch_ds(model, x, y, dataset)
    return clientloaders, valloader, testloader, test_batch

def partition_iidmix(client_lens, p):
    total_lens = np.cumsum(client_lens)
    total_lens = total_lens - total_lens[0]
    clients = [ # keep first (1-p)*client_len examples
        np.arange(curr_idx,int(curr_idx+(1-p)*client_len)) for 
        client_len,curr_idx in zip(client_lens, total_lens)
    ]
    pool_idx = np.concatenate([
        # pool last p*client_len examples
        np.arange(int(curr_idx+(1-p)*client_len), curr_idx+client_len) for 
        client_len,curr_idx in zip(client_lens, total_lens)
    ])
    pool_idx = pool_idx[np.random.permutation(len(pool_idx))] # random shuffle
    S = int(len(pool_idx) / len(client_lens))
    for i,keep_idx in enumerate(clients):
        clients[i] = np.concatenate((keep_idx, pool_idx[S*i:S*(i+1)]))
    return clients

def partition_dirichlet(Y, n_clients, alpha, seed):
    clients = []
    ex_per_class = np.unique(Y, return_counts=True)[1]
    n_classes = len(ex_per_class)
    print(f"Found {n_classes} classes")
    rv_tr = stats.dirichlet.rvs(np.repeat(alpha, n_classes), size=n_clients, random_state=seed) 
    rv_tr = rv_tr / rv_tr.sum(axis=0)
    rv_tr = (rv_tr*ex_per_class).round().astype(int)
    class_to_idx = {i: np.where(Y == i)[0] for i in range(n_classes)}
    curr_start = np.zeros(n_classes).astype(int)
    for client_classes in rv_tr:
        curr_end = curr_start + client_classes
        client_idx = np.concatenate([class_to_idx[c][curr_start[c]:curr_end[c]] for c in range(n_classes)])
        curr_start = curr_end
        clients.append(client_idx)
        # will be empty subset if all examples have been exhausted
    return clients

def build_flair(eval_frac=1):
    fp = FLAIRPooled()
    return fp.get_split('train', pooled=False), fp.get_split('val', pooled=True, subsample=eval_frac), fp.get_split('test', pooled=True)
    # return FLAIRClients(), fp.get_split('val', pooled=True), fp.get_split('test', pooled=True)


def build_cifar10(n_clients, alpha, seed):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=True, download=True, transform=transform)
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = np.array([trainset.targets[i] for i in trainidx])
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    validx = np.arange(int(N*0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)
    testset = torchvision.datasets.CIFAR10(root=f"{DATA}/cifar10", train=False, download=True, transform=test_transform)
    return clients, valset, testset

def build_20newsgroups(n_clients, alpha, seed):
    train_pt = f"{DATA}/20newsgroups/20newsgroups_train.pt"
    test_pt = f"{DATA}/20newsgroups/20newsgroups_test.pt"
    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_20newsgroups_dump()
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    trainset = list(zip(tr_d['X'], tr_d['Y']))
    testset = list(zip(ev_d['X'], ev_d['Y']))
    N = len(trainset)
    trainidx = np.arange(0, int(N*0.8))
    Y_tr = tr_d['Y'][trainidx]
    clientidx = partition_dirichlet(Y_tr, n_clients, alpha, seed)
    clients = [torch.utils.data.Subset(trainset, trainidx[cidx]) for cidx in clientidx]
    validx = np.arange(int(N*0.8), N)
    valset = torch.utils.data.Subset(trainset, validx)
    return clients, valset, testset

def generate_20newsgroups_dump():
    print("Generating 20newsgroups cache...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = 50256
    ng_train = fetch_20newsgroups(subset='train')
    tr_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_train['data']])

    ng_test = fetch_20newsgroups(subset='test')
    ev_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_test['data']])

    tr_Y = torch.LongTensor(ng_train['target'])
    ev_Y = torch.LongTensor(ng_test['target'])

    os.makedirs(f"{DATA}/20newsgroups", exist_ok=True)
    torch.save({'X': tr_X, 'Y': tr_Y}, f"{DATA}/20newsgroups/20newsgroups_train.pt")
    torch.save({'X': ev_X, 'Y': ev_Y}, f"{DATA}/20newsgroups/20newsgroups_test.pt")

def build_reddit(alpha, seed):
    train_X = []
    with open(f"{DATA}/reddit/train_clients.json") as f:
        client_names = json.load(f)
        for client_name in client_names.keys():
            client_X = torch.load(f"{DATA}/reddit/cache/{client_name}.pt")['X']
            train_X.append(client_X)
    trainlen = int(len(train_X)*0.8)
    
    eval_X = train_X[trainlen:]
    eval_X = torch.cat(eval_X)
    eval_Y = [-1 for i in range(len(eval_X))]
    evalset = list(zip(eval_X,eval_Y))
    train_X[:trainlen]

    test_X = []
    with open(f"{DATA}/reddit/eval_clients.json") as f:
        client_names = json.load(f)
        for client_name in client_names.keys():
            client_X = torch.load(f"{DATA}/reddit/cache/{client_name}.pt")['X']
            test_X.append(client_X)
    test_X = torch.cat(test_X)
    test_Y = [-1 for i in range(len(test_X))]
    testset = list(zip(test_X,test_Y))

    assert 0 <= alpha <= 1, "For Reddit, set iid_alpha >= 0 (non-iid default) and <= 1 (iid)"
    client_idx = partition_iidmix([len(X) for X in train_X], alpha)
    train_X_flat = torch.cat(train_X)
    clients = [[(train_X_flat[i],-1) for i in idx] for idx in client_idx]
    return clients, evalset, testset

class FLAIRClients(torch.utils.data.Dataset):
    def __init__(self):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229 ** 2, 0.224 ** 2, 0.225 ** 2)) # imagenet
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        with open(f"{DATA}/ml-flair/data/labels_and_metadata.json") as f:
            self.data = json.load(f)
        self.labelset = {l:i for i, l in enumerate(set(l for d in self.data for l in d['labels']))}
        def get_one_hot(label_names):
            y = torch.zeros(len(self.labelset))
            for l in label_names:
                y[self.labelset[l]] = 1
            return y
        self.Y = torch.stack([get_one_hot(d['labels']) for d in self.data])

        self.client2idx = {}
        self.client2split = {}
        for i,d in enumerate(self.data):
            if d['partition'] == 'train':
                if d['user_id'] not in self.client2idx:
                    self.client2idx[d['user_id']] = []
                    self.client2split[d['user_id']] = d['partition']
                self.client2idx[d['user_id']].append(i)
                assert self.client2split[d['user_id']] == d['partition']
        self.gid2cid = {i:k for i,k in enumerate(self.client2idx.keys())}

    def __len__(self):
        return len(self.client2idx)

    def __getitem__(self, cid):
        x,y = [], []
        cidx = self.client2idx[self.gid2cid[cid]]
        for idx in cidx:
            path = f"{DATA}/ml-flair/data/small_images/{self.data[idx]['image_id']}.jpg"
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            sample = self.transform(img)
            x.append(sample)
            y.append(self.Y[idx])
        return torch.stack(x), torch.stack(y)
    
class FLAIRPooled(torch.utils.data.Dataset):
    def __init__(self):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229 ** 2, 0.224 ** 2, 0.225 ** 2)) # imagenet
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        with open(f"{DATA}/ml-flair/data/labels_and_metadata.json") as f:
            self.data = json.load(f)
        
        self.labelset = {l:i for i, l in enumerate(set(l for d in self.data for l in d['labels']))}
        def get_one_hot(label_names):
            y = torch.zeros(len(self.labelset))
            for l in label_names:
                y[self.labelset[l]] = 1
            return y
        self.Y = torch.stack([get_one_hot(d['labels']) for d in self.data])

        self.client2idx = {}
        self.client2split = {}
        for i,d in enumerate(self.data):
            if d['user_id'] not in self.client2idx:
                self.client2idx[d['user_id']] = []
                self.client2split[d['user_id']] = d['partition']
            self.client2idx[d['user_id']].append(i)
            assert self.client2split[d['user_id']] == d['partition']
    
    def get_split(self, split, pooled, subsample=1):        
        if pooled:
            pooled_idx = np.concatenate([
                client_idx for client_id, client_idx in self.client2idx.items() 
                if self.client2split[client_id] == split])
            if subsample < 1:
                pooled_idx = pooled_idx[:int(len(pooled_idx)*subsample)]
            data = torch.utils.data.Subset(self, pooled_idx)
        else:
            data = [torch.utils.data.Subset(self, client_idx) for client_id, client_idx 
                    in self.client2idx.items() if self.client2split[client_id] == split]
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = f"{DATA}/ml-flair/data/small_images/{self.data[idx]['image_id']}.jpg"
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        sample = self.transform(img) if self.transform else img
        target = self.Y[idx]
        return sample, target