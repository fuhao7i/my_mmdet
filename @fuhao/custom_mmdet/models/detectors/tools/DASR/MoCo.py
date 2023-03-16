# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from .vgg import VGG

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 256),
        # )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        # out = self.mlp(fea)
        out = fea
        return fea, out


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=128, K=32*256, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = Encoder()
        # self.encoder_k = Encoder()
        
        self.encoder_q = VGG()

        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        flag = (ptr + batch_size) % self.K
        # print(batch_size, ptr, flag)
        if flag < ptr + batch_size:
            # print(' spatial ', flag)
            keys = keys.transpose(0, 1)
            # print(keys.shape)
            # print(self.queue.shape)
            self.queue[:, ptr:self.K] = keys[:, :batch_size - flag]
            ptr = 0
            self.queue[:, ptr:ptr + flag] = keys[:, batch_size - flag:]
            ptr = flag
            
        else:
            # assert self.K % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k1, im_k2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.training:
            # compute query features

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self.encoder_q.eval()
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)

                k1 = self.encoder_q(im_k1)  # keys: NxC
                k1 = nn.functional.normalize(k1, dim=1)

                k2 = self.encoder_q(im_k2)  # keys: NxC
                k2 = nn.functional.normalize(k2, dim=1)
                self._dequeue_and_enqueue(k2)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            # print(q.shape, k1.shape)
            l_pos = torch.einsum('nc,nc->n', [q, k1]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            # self._dequeue_and_enqueue(k2)

            return q, logits, labels
        else:
            embedding, _ = self.encoder_q(im_q)

            return embedding


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
