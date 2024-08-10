import torch
import torch.nn as nn
from torchattacks.attack import Attack

import numpy as np

class PGD(Attack):
        #attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        #adv_images = attack(images, labels)

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, seq, labels):
        seq = seq.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(seq, labels)

        loss = nn.CrossEntropyLoss()
        adv_seq = seq.clone().detach()

        if self.random_start:
            adv_seq = adv_seq + torch.empty_like(adv_seq).uniform_(
                -self.eps, self.eps
            )
            adv_seq = adv_seq.detach()

        for _ in range(self.steps):
            adv_seq.requires_grad = True
            outputs = self.get_logits(adv_seq)

            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_seq, retain_graph=False, create_graph=False
            )[0]

            adv_seq = adv_seq.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_seq - seq, min=-self.eps, max=self.eps)
            adv_seq = (seq + delta).detach()

        return adv_seq