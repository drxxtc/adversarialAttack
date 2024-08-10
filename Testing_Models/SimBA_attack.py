import torch
import torch.nn as nn
import torch.nn.functional as F

class SimBA:

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def get_probs(self, x, y):
        output = pred = self.model(x)
        probs = torch.index_select(F.softmax(output.detach(), dim=1), 1,torch.tensor([y]))
        return probs

    def get_preds(self, x):
        output = pred = self.model(x)
        pred = F.softmax(output.detach(), dim=1).numpy().argmax(1)
        return pred

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, epsilon, num_iters, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_prob = self.get_probs(x, y)

        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon

            left_prob = self.get_probs((x - diff.view(x.size())), y)
            if left_prob < last_prob:
                x = (x - diff.view(x.size()))
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())), y)
                if right_prob < last_prob:
                    x = (x + diff.view(x.size()))
                    last_prob = right_prob

        success = (last_prob <= 1/2)
        return x, int(success)