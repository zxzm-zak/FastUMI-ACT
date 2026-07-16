import os
import sys

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(CURRENT_DIR)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL weight: {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        image = normalize(image)

        if actions is None:
            _, a_hat_gru, _, _ = self.model(qpos, image, env_state)
            return a_hat_gru

        actions = actions[:, : self.model.num_queries]
        is_pad = is_pad[:, : self.model.num_queries]
        a_hat, a_hat_gru, _, (mu, logvar) = self.model(
            qpos, image, env_state, actions, is_pad
        )
        total_kld, _, _ = kl_divergence(mu, logvar)
        kl = total_kld[0]

        all_l1_main = F.l1_loss(a_hat, actions, reduction="none")
        l1_main = (all_l1_main * ~is_pad.unsqueeze(-1)).mean()
        all_l1_gru = F.l1_loss(a_hat_gru, actions, reduction="none")
        l1_gru = (all_l1_gru * ~is_pad.unsqueeze(-1)).mean()

        return {
            "l1_main": l1_main,
            "l1_gru": l1_gru,
            "kl": kl,
            "loss": l1_main + l1_gru + kl * self.kl_weight,
        }

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        image = normalize(image)
        if actions is None:
            return self.model(qpos, image, env_state)

        actions = actions[:, 0]
        a_hat = self.model(qpos, image, env_state, actions)
        mse = F.mse_loss(actions, a_hat)
        return {"mse": mse, "loss": mse}

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
