# flasc

This repository contains code for "[Federated LoRA with Sparse Communication](https://arxiv.org/abs/2406.05233)" .

We use PyTorch and Huggingface Transformers to simulate federated training of LoRA. To begin, install CUDA 11 or later, Python 3.10.14, and Anaconda / Miniconda on the host machine.

# Envrionment Setup
```
conda create --name flasc python=3.10.14
```

We install all the packages manually using pip.

```
conda activate flasc
```

```
python -m pip install tqdm scikit_learn numpy scipy torch torchvision torchaudio tensorflow_cpu peft transformers
```


The code provides utilities to automatically download the CIFAR10 and 20NewsGroups datasets. We will provide full details for working with the other datasets once the paper is made public.

To train a global LoRA module, run the ```train_sparse_lora_dp.py``` script.

To reproduce the results on systems heterogeneity, we provide the script ```train_sparse_lora_het.py```. The script hard-codes tiers of per-round communication capability in powers of 4.