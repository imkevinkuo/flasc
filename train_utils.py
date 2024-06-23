import torch
import tensorflow as tf

sig = torch.nn.Sigmoid()

def test_batch_cls(model, x, y, multilabel=False): # classification
    outputs = model(x, labels=y)
    logits = outputs.logits.detach()
    loss = outputs.loss # huggingface loss is already averaged
    if multilabel: # label set is a binary vector
        preds = torch.where(sig(logits) < 0.5, 0, 1)
        stats = {
            'tp': (preds*y).sum().item(),
            'tn': ((1-preds)*(1-y)).sum().item(),
            'fp': (preds*(1-y)).sum().item(),
            'fn': ((1-preds)*y).sum().item(),
            'count': x.shape[0],
            'loss': loss.item()*x.shape[0],
        }
    else: # labels are integers
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        stats = {
            'tp': correct,
            'fp': len(y) - correct,
            'fn': len(y) - correct,
            'count': x.shape[0],
            'loss': loss.item()*x.shape[0],
        }
    return loss, stats

def test_batch_nwp(model, x): # next word (token) prediction
    non_pad_idx = x[:, 1:] != 50256                       # [B, S]: bool
    total = non_pad_idx.sum().item()                      # [sentences]: int
    output = model(x)
    logits = output.logits[:, :-1]
    flat_logits = logits.reshape(-1, 50257) # exclude last token
    loss = torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(flat_logits, dim=-1), # flat predictions
        x[:, 1:].reshape(-1), # flat tokens
        ignore_index=50256,
        reduction='sum') / total
    with torch.no_grad():
        pred_toks = logits.argmax(dim=-1)                 # [sentences, tokens]: 0...50256
        correct_toks = pred_toks == x[:, 1:]              # [sentences, tokens]: bool
        correct = (non_pad_idx*correct_toks).sum().item() # [sentences]: int
        stats = {
            'tp': correct, 
            'fp': total - correct, 
            'fn': total - correct,
            'count': total,
            'loss': loss.item()*total,
        }
    return loss, stats

def get_metric(stats, metric):
        if stats['tp'] == 0:
            return 0
        elif metric == 'accu':
            return stats['tp'] / (stats['tp'] + stats['fp'])
        elif metric == 'recall':
            return stats['tp'] / (stats['tp'] + stats['fn'])
        elif metric == 'f1':
            return 2*stats['tp'] / (2*stats['tp'] + stats['fp'] + stats['fn'])
    
def log_stats(writer, prefix, stats, step):
    with writer.as_default():
        tf.summary.scalar(f"{prefix}/accuracy", get_metric(stats, 'accu'), step=step)
        tf.summary.scalar(f"{prefix}/recall", get_metric(stats, 'recall'), step=step)
        tf.summary.scalar(f"{prefix}/f1", get_metric(stats, 'f1'), step=step)
        for k,v in stats.items():
            if k not in ['tp', 'fp', 'tn', 'fn']:
                tf.summary.scalar(f"{prefix}/{k}", v, step=step)