import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply
from torch.autograd import Variable
from torch import FloatTensor
from torch.distributions.normal import Normal


## For k = 1 we need to use Gumbel-softmax or sth

class SparseGate(nn.Module):
    def __init__(self, in_features, n_experts, k=2):
        '''
        Returns a sparsely gated noisy softmax.
        See OUTRAGEOUSLY LARGE NEURAL NETWORKS:
            THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER
            Shazeer et. al
            Link: https://arxiv.org/pdf/1701.06538.pdf
        '''
        assert k > 1, "Need k >= 1. If k == 1, then derivatives are zero everywhere."
        super(SparseGate, self).__init__()
        self.gate_weights = Parameter(torch.Tensor(n_experts, in_features))
        self.noise_weights = Parameter(torch.Tensor(n_experts, in_features))
        self.n_experts = n_experts
        self.n_selected = k
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.size(0)
        noise = x.new_empty((batch_size, self.n_experts)).normal_()
        expert_weights = F.linear(x, self.gate_weights, None) + noise * F.softplus(
            F.linear(x, self.noise_weights, None))  # self.noise_weights(x))
        top_k, indices = torch.topk(expert_weights, self.n_selected)
        top_k_softmax = F.softmax(top_k, dim=1)
        res = x.new_full((batch_size, self.n_experts), 0.0)
        return res.scatter_(1, indices, top_k_softmax)

    def reset_parameters(self):
        nn.init.constant_(self.gate_weights, 0.0)
        nn.init.constant_(self.noise_weights, 0.0)


class SparselyGatedMoELayer(nn.Module):

    def __init__(self, in_features, experts, k=2, use_parallel_apply=False, temperature=1.0):
        '''
        Returns the outputs of a sparsely gated mixture of experts

        inputs:
            experts:
                A list of models which accept the same input and produce the same output

        returns:
            An output which is the sum of the outputs from the experts

        See OUTRAGEOUSLY LARGE NEURAL NETWORKS:
            THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER
            Shazeer et. al
            Link: https://arxiv.org/pdf/1701.06538.pdf
        '''
        super(SparselyGatedMoELayer, self).__init__()
        if k == 1:
            self.sparse_gate = STGumbelSoftmax(in_features, len(experts), k, temperature)
        else:
            self.sparse_gate = SparseGate(in_features, len(experts), k)
        self.experts = experts
        self.n_selected = k
        self._parallel_apply = use_parallel_apply
        self._parallel_sum = False
        self.print_gates = False
        self.last_experts = None

    def forward(self, x_gate, x_experts):
        x = self.sparse_gate(x_gate)
        selected_experts = (x != 0).nonzero()  # ret: tuple of x's, y's, z's (indices) where x is not 0
        inputs_for_experts = []
        batch_indices_for_experts = []

        for i in range(len(self.experts)):
            expert_was_selected = selected_experts[:, 1] == i
            batch_index_for_expert = selected_experts[expert_was_selected, 0]
            batch_indices_for_experts.append(batch_index_for_expert)
            inputs = None if len(batch_index_for_expert) == 0 else x_experts[batch_index_for_expert]
            inputs_for_experts.append(inputs)

        experts_to_run = []
        inputs_to_feed = []
        batch_indices_to_scatter = []
        expert_run_to_orig_index = []
        for i, (expert, inputs, batch_index) in enumerate(
            zip(self.experts, inputs_for_experts, batch_indices_for_experts)):
            if len(batch_index) > 0:
                experts_to_run.append(expert)
                inputs_to_feed.append(inputs.unsqueeze(0))
                batch_indices_to_scatter.append(batch_index)
                expert_run_to_orig_index.append(i)

        if self._parallel_apply:
            res = parallel_apply(experts_to_run, inputs_to_feed)
            # If the number of selected experts is very large then we can do it in parallel
            if self._parallel_sum:
                def scatter_batch(r, indices_to_scatter, i):
                    output = x.new_full((x_gate.shape[0],) + r.shape[1:], 0)
                    attention = x[(indices_to_scatter, expert_run_to_orig_index[i]) + (None,) * (len(r.shape) - 1)]
                    output[indices_to_scatter] += attention * r
                    return output

                output_ = parallel_apply([scatter_batch] * len(res),
                                         [(r, idx, i) for i, (r, idx) in enumerate(zip(res, batch_indices_to_scatter))])
                output = torch.sum(torch.stack(output_, dim=0), dim=0)
            else:
                output = x.new_full((x_gate.shape[0],) + res[0].shape[1:], 0)
                for i, (indices_to_scatter, r) in enumerate(zip(batch_indices_to_scatter, res)):
                    attention = x[(indices_to_scatter, expert_run_to_orig_index[i]) + (None,) * (len(r.shape) - 1)]
                    output[indices_to_scatter] += attention * r
        else:
            res = []
            for expert, inputs in zip(experts_to_run, inputs_to_feed):
                res.append(expert(inputs.squeeze(0)))
            output = x.new_full((x_gate.shape[0],) + res[0].shape[1:], 0)
            for i, (indices_to_scatter, r) in enumerate(zip(batch_indices_to_scatter, res)):
                attention = x[(indices_to_scatter, expert_run_to_orig_index[i]) + (None,) * (len(r.shape) - 1)]
                output[indices_to_scatter] += attention * r

        self.last_experts = x.cpu().detach()
        self.output_mean = output.mean().cpu().detach()
        # self.output_std = output.mean().cpu().detach()
        return output


class STGumbelSoftmax(nn.Module):
    def __init__(self, in_features, n_experts, k=2, temperature=1.0):
        '''
        Returns a sparsely gated noisy softmax.
        See OUTRAGEOUSLY LARGE NEURAL NETWORKS:
            THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER
            Shazeer et. al
            Link: https://arxiv.org/pdf/1701.06538.pdf
        '''
        assert k == 1, "Need k == 1. If k == 1, then derivatives are zero everywhere."
        super().__init__()
        self.gate_weights = Parameter(torch.Tensor(n_experts, in_features))
        self.eps = 1e-9
        self.temperature = temperature
        self.n_experts = n_experts
        self.n_selected = k
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.size(0)
        expert_logits = F.linear(x, self.gate_weights, None)
        return st_gumbel_softmax(expert_logits, self.temperature)

    def reset_parameters(self):
        nn.init.constant_(self.gate_weights, 0.0)


"""
From https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
"""


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def st_gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y