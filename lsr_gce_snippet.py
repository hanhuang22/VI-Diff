import torch

def label_smoothing(outputs, labels, epsilon=0.1):
    num_classes = outputs.size(1)
    smoothed_labels = torch.full_like(outputs, epsilon / (num_classes - 1))
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - epsilon)
    return smoothed_labels

### credit to: https://github.com/DSXiangLi/ClassicSolution/blob/main/src/loss.py
class GeneralizeCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizeCrossEntropy, self).__init__()
        self.q = q

    def forward(self, logits, labels):
        # Negative box cox: (1-f(x)^q)/q
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        loss = (1 - torch.pow(torch.sum(labels * probs, dim=-1), self.q) )/ self.q
        loss = torch.mean(loss)
        return loss
