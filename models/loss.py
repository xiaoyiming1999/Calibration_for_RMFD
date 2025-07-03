import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Confidence(nn.Module):
    def __init__(self):
        super(Confidence, self).__init__()

    def forward(self, outputs):
        outputs = F.softmax(outputs, dim=1)
        log_outputs = torch.log2(outputs + 1e-8)
        loss = torch.sum(-outputs * log_outputs, dim=-1)
        loss = loss.mean()

        return loss


class Focal_loss(nn.Module):
    def __init__(self, num_class, gamma=1):
        super(Focal_loss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.alpha = torch.ones(self.num_class, 1)

    def forward(self, outputs, labels):
        outputs = F.softmax(outputs, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了
        labels = labels.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha.to(outputs.device)
        gamma = self.gamma

        idx = labels.cpu().long()
        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(outputs.device)

        pt = (one_hot_key * outputs).sum(1) + epsilon
        log_pt = pt.log()
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * log_pt
        loss = loss.mean()

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, num_cls, smoothing=0.05):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.num_cls = num_cls

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        logprobs = F.log_softmax(x, dim=-1)
        target = F.one_hot(target, self.num_cls)
        target = (1.0 - self.smoothing) * target + self.smoothing / self.num_cls
        target = torch.clamp(target.float(), min=self.smoothing / (self.num_cls - 1), max=1.0 - self.smoothing)
        loss = -1 * torch.sum(target * logprobs, 1)

        return loss.mean()


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class BS_loss(nn.Module):

    def __init__(self, num_cls):
        """
        n_bins (int): number of confidence interval bins
        """
        super(BS_loss, self).__init__()
        self.num_class = num_cls

    def forward(self, outputs, labels):
        softmax_outputs = outputs.softmax(dim=-1)
        labels = labels.view(-1, 1)

        idx = labels.cpu().long()
        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(outputs.device)

        loss = torch.mean(torch.sum((softmax_outputs - one_hot_key) ** 2, dim=-1))

        return loss

def compute_entropy(outputs):

    softmax_outputs = outputs.softmax(dim=-1)
    log_outputs = torch.log(softmax_outputs + 1e-10)
    entropy = torch.sum(-softmax_outputs * log_outputs, dim=-1)

    return entropy

def compute_fpr_at_95tpr(y_true, entropy_scores):
    """
    computer FPR when TPR=95%
    :param y_true: true label
    :param entropy_scores: uncertainty
    :return: FPR@95TPR
    """

    pos_scores = entropy_scores[y_true == 1]
    neg_scores = entropy_scores[y_true == 0]

    P = len(pos_scores)
    N = len(neg_scores)

    target_tp = np.ceil(0.9 * P).astype(int)

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(P), np.zeros(N)])

    sorted_indices = np.argsort(-all_scores)
    sorted_labels = all_labels[sorted_indices]

    # 累计TP和FP
    tp, fp = 0, 0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1

        if tp >= target_tp:
            return fp / N

    return np.nan