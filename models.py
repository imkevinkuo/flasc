from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

def build_model(dataset):
    if dataset == 'cifar10':
        model_name = "google/vit-base-patch16-224-in21k"
        model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=10)
    elif dataset == 'flair':
        model_name = "google/vit-base-patch16-224-in21k"
        # model_name = "microsoft/resnet-18"
        model = AutoModelForImageClassification.from_pretrained(
            model_name, 
            num_labels=17, 
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True)
        # for n,p in model.named_parameters():
        #     if 'normalization' in n:
        #         p.requires_grad = False
        # torch.nn.init.kaiming_normal_(model.classifier[1].weight.data)
        # model.classifier[1].bias.data *= 0
    elif dataset == '20newsgroups':
        model_name = "gpt2"
        model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=20,pad_token_id=50256)
    elif dataset == 'reddit':
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name,pad_token_id=50256)
    return model

def add_adapters_dataset(dataset, model, lora_rank, lora_alpha):
    if dataset == 'cifar10':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
    elif dataset == 'flair':
        add_adapters(model, lora_rank, lora_alpha, "classifier", ["query", "value"])
        # add_adapters(model, lora_rank, lora_alpha, 'classifier', ['convolution'])
    elif dataset == '20newsgroups':
        add_adapters(model, lora_rank, lora_alpha, "score", ["c_attn", "c_proj", "c_fc"])
    elif dataset == 'reddit':
        add_adapters(model, lora_rank, lora_alpha, None, ["c_attn", "c_proj", "c_fc"])

def add_adapters(model, lora_rank, lora_alpha, output_layer_name, target_modules):
    from peft import LoraConfig, get_peft_model

    if lora_rank > 0:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[output_layer_name] if output_layer_name is not None else [],
        )
        model = get_peft_model(model, config)
    elif lora_rank == 0: # linear fine-tune
        for n,p in model.named_parameters():
            if output_layer_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    # if lora_rank == -1: full fine-tune