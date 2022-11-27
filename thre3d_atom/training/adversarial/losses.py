import abc
from typing import Tuple, Dict, Any, List

import torch
from thre3d_atom.networks.shared.utils import detach_tensor_from_graph
from thre3d_atom.training.adversarial.models import Discriminator
from torch import Tensor
from torch import autograd
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss


# noinspection PyUnresolvedReferences
def _compute_grad2(d_outs: List[Tensor], x_in: Tensor) -> Tensor:
    """computes squared gradient penalty"""
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.shape[0]
        grad_dout = autograd.grad(
            outputs=d_out.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert grad_dout2.shape == x_in.shape
        reg += grad_dout2.reshape(batch_size, -1).sum(1)
    return reg / len(d_outs)


class GanLoss(metaclass=abc.ABCMeta):
    # This involves a lot of magic, but provides great flexibility for debugging
    ExtraGanLossInfo = Dict[str, Any]

    @abc.abstractmethod
    def dis_loss(
        self, discriminator: Discriminator, real: Tensor, fake: Tensor, **kwargs
    ) -> Tuple[Tensor, ExtraGanLossInfo]:
        """
        Computes the discriminator loss given the real and fake tensors
        Args:
            discriminator: discriminator object
            real: real examples
            fake: fake examples
            **kwargs: additional inputs
        Returns: computed loss, additional information
        """

    @abc.abstractmethod
    def gen_loss(
        self, discriminator: Discriminator, real: Tensor, fake: Tensor, **kwargs
    ) -> Tuple[Tensor, ExtraGanLossInfo]:
        """
        Computes the generator loss given the real and fake tensors
        Args:
            discriminator: discriminator object
            real: real examples
            fake: fake examples
            **kwargs: additional inputs
        Returns: computed loss, additional information
        """


class StandardGanLoss(GanLoss):
    def __init__(self, real_gradient_penalty: bool = True, penalty_scale: float = 1.0):
        self._criterion = BCEWithLogitsLoss()
        self._rgp = real_gradient_penalty
        self._penalty_scale = penalty_scale

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        if self._rgp:
            # add real_samples to the graph
            real_samples.requires_grad_(True)

        real_scores, _ = discriminator(real_samples)
        fake_scores, _ = discriminator(detach_tensor_from_graph(fake_samples))

        real_loss = self._criterion(
            real_scores, torch.ones(real_scores.shape).to(real_scores.device)
        )
        fake_loss = self._criterion(
            fake_scores, torch.zeros(fake_scores.shape).to(fake_scores.device)
        )
        normal_loss = (real_loss + fake_loss) / 2

        # compute the real (one-sided) gradient penalty
        penalty, real_disc_gradient_norm = (
            torch.zeros(1, device=real_scores.device),
            torch.zeros(1, device=real_scores.device),
        )
        if self._rgp:
            real_disc_gradient_norm = _compute_grad2(real_scores, real_samples)
            penalty = (0.5 * self._penalty_scale * real_disc_gradient_norm).mean()
        loss = normal_loss + penalty

        return loss, {
            "discriminator_loss_value": normal_loss,
            "discriminator_real_gradient_norm": real_disc_gradient_norm.mean(),
            "dis_real_scores": real_scores.mean(),
            "dis_fake_scores": fake_scores.mean(),
        }

    def gen_loss(
        self, discriminator: Discriminator, _: Tensor, fake_samples: Tensor, **kwargs
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        fake_scores, _ = discriminator(fake_samples)
        return (
            self._criterion(
                fake_scores, torch.ones(fake_scores.shape).to(fake_scores.device)
            ),
            {},
        )


class PairedAngularGanLoss(GanLoss):
    @staticmethod
    def _compute_angular_discrepancy(
        real_embeddings: List[Tensor], fake_embeddings: List[Tensor]
    ) -> Tensor:
        dot_products_list = []
        for real_embedding, fake_embedding in zip(real_embeddings, fake_embeddings):
            # normalize the real and fake embeddings:
            real_embedding_normalized = real_embedding / real_embedding.norm(
                dim=1, keepdim=True
            )
            fake_embedding_normalized = fake_embedding / fake_embedding.norm(
                dim=1, keepdim=True
            )

            # compute the angular discrepancy
            dot_products = torch.sum(
                real_embedding_normalized * fake_embedding_normalized, dim=1
            )
            dot_products_list.append(torch.mean(dot_products))
        return torch.mean(torch.stack(dot_products_list))

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        _, real_embeddings = discriminator(real_samples)
        _, fake_embeddings = discriminator(detach_tensor_from_graph(fake_samples))
        return -self._compute_angular_discrepancy(real_embeddings, fake_embeddings), {}

    def gen_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        _, real_embeddings = discriminator(real_samples)
        _, fake_embeddings = discriminator(fake_samples)
        return self._compute_angular_discrepancy(real_embeddings, fake_embeddings), {}


class EmbeddingLatentGanLoss(GanLoss):
    def __init__(self):
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        real_scores, _ = discriminator(real_samples)
        fake_scores, _ = discriminator(detach_tensor_from_graph(fake_samples))

        real_loss = self.criterion(
            real_scores, torch.ones(real_scores.shape).to(real_scores.device)
        )
        fake_loss = self.criterion(
            fake_scores, torch.zeros(fake_scores.shape).to(fake_scores.device)
        )
        return (real_loss + fake_loss) / 2, {}

    def gen_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        _, real_embeddings = discriminator(real_samples)
        _, fake_embeddings = discriminator(fake_samples)
        embedding_loss_list = []
        for real_embedding, fake_embedding in zip(real_embeddings, fake_embeddings):
            embedding_loss_list.append(l1_loss(fake_embedding, real_embedding))
        return torch.mean(torch.stack(embedding_loss_list)), {}


class WganGPGanLoss(GanLoss):
    def __init__(self, gp_lambda: float = 0.1, drift_lambda: float = 0.0) -> None:
        self._drift_lambda = drift_lambda
        self._gp_lambda = gp_lambda

    @staticmethod
    def _gradient_penalty(
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        reg_lambda: float = 0.1,
    ) -> Tensor:
        """
        private helper for calculating the gradient penalty
        Args:
            discriminator: the discriminator used for computing the penalty
            real_samples: real samples
            fake_samples: fake samples
            reg_lambda: regularisation lambda
        Returns: computed gradient penalty
        """
        batch_size = real_samples.shape[0]

        # generate random epsilon
        epsilon = torch.rand(
            batch_size, *(1 for _ in range(len(real_samples.shape[1:])))
        ).to(real_samples.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        dis_scores, _ = discriminator(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=dis_scores,
            inputs=merged,
            grad_outputs=torch.ones_like(dis_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # calculate the penalty using these gradients
        mean_gradient_norm = (gradient.norm(p=2, dim=1)).mean()
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty, mean_gradient_norm

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        fake_samples = detach_tensor_from_graph(fake_samples)
        real_scores, _ = discriminator(real_samples)
        fake_scores, _ = discriminator(fake_samples)

        normal_loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self._drift_lambda * torch.mean(real_scores ** 2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gradient_penalty, mean_gradient_norm = self._gradient_penalty(
            discriminator, real_samples, fake_samples, reg_lambda=self._gp_lambda
        )
        loss = normal_loss + gradient_penalty

        key_suffix = kwargs.get("suffix", "")
        return loss, {
            f"discriminator_loss_value_{key_suffix}": normal_loss,
            f"discriminator_real_gradient_norm_{key_suffix}": mean_gradient_norm,
            f"dis_real_scores_{key_suffix}": real_scores.mean(),
            f"dis_fake_scores_{key_suffix}": fake_scores.mean(),
        }

    def gen_loss(
        self, discriminator: Discriminator, _: Tensor, fake_samples: Tensor, **kwargs
    ) -> Tuple[Tensor, GanLoss.ExtraGanLossInfo]:
        fake_scores, _ = discriminator(fake_samples)
        return -torch.mean(fake_scores), {}
