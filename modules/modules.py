from torch import nn
import torch
from torch.distributions import Normal
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.nn.functional as F
from modules.sparsemax import *
# from modules.densenet import *

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=1):
        super(ProjectionHead, self).__init__()

        self.head = nn.Sequential()
        for i in range(nlayers - 1):
            self.head.add_module(f"linear_{i}", nn.Linear(in_dim, in_dim))
            self.head.add_module(f"relu_{i}", nn.ReLU())
        self.head.add_module(f"linear_final", nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.head(x)

class MultiHeadLinearProjection(nn.Module):
    def __init__(self, output_size, in_dim, nlayers=1):
        super(MultiHeadLinearProjection, self).__init__()
        self.linears = nn.ModuleList()
        self.output_size = output_size
        self.in_dim = in_dim
        for i in output_size:
            self.linears.append(ProjectionHead(in_dim, i, nlayers))

    def forward(self, features):
        out = []
        for head in self.linears:
            out += [head(features) / (features.shape[1]  ** 0.5)]
        return out

class HyperNet(nn.Module):
    def __init__(self, cfg):
        super(HyperNet, self).__init__()

        self.hidden_layer_sizes = [cfg.hypernet_hidden_size] * cfg.hypernet_num_hidden_layers
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(cfg.hypernet_input_dim, self.hidden_layer_sizes[0])]
        self.num_joints = cfg.num_joints

        for i in range(len(self.hidden_layer_sizes) - 1):
            self.layers += [nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1])]

        self.out = nn.Linear(self.hidden_layer_sizes[-1], cfg.embedding_dim)

        num_parameters_lst = []
        for i in range(1, cfg.num_joints + 1):
            num_parameters_lst += [cfg.jointnet_hidden_size * i,
                                   cfg.jointnet_hidden_size ,
                                   cfg.jointnet_hidden_size  * cfg.jointnet_output_dim,
                                   cfg.jointnet_output_dim]

        self.projection = MultiHeadLinearProjection(num_parameters_lst, cfg.embedding_dim , 1)
        self.nonlinearity = torch.relu

        self.bns = nn.ModuleList()
        self.bns += [torch.nn.BatchNorm1d(cfg.hypernet_input_dim)]
        self.bns +=[torch.nn.BatchNorm1d(dim) for dim in self.hidden_layer_sizes]

    def forward(self, inputs):
        x = inputs
        for k, layer in enumerate(self.layers):
            x = self.bns[k](x)
            x = self.nonlinearity(layer(x))
        features = self.out(x)
        weights = self.projection(features)
        return weights

class JointNetTemplate(nn.Module):
    def __init__(self, cfg):
        super(JointNetTemplate, self).__init__()
        self.hidden_layer_size = cfg.jointnet_hidden_size
        self.output_dim = cfg.jointnet_output_dim


    def forward(self, inp, weights):
        if inp.shape[1] == 1:
            out = inp * weights[0] + weights[1]
        else:
            out = torch.bmm(inp.unsqueeze(1), weights[0].reshape(weights[0].shape[0], inp.shape[1], self.hidden_layer_size)).squeeze(1) + weights[1]
        out = torch.relu(out)
        out = torch.bmm(out.unsqueeze(1), weights[2].reshape(weights[2].shape[0],self.hidden_layer_size, self.output_dim)).squeeze(1) + weights[3]
        return out


class MainNet(nn.Module):
    def __init__(self, cfg):
        super(MainNet, self).__init__()
        self.num_joints = cfg.num_joints
        self.num_gaussians = cfg.num_gaussians
        self.joint_template = JointNetTemplate(cfg)


    def forward(self, x, weights):
        out = []
        x = x.unsqueeze(2)
        for i in range(x.shape[1]-1):
            out.append(self.joint_template(x[:, :i + 1].squeeze(2), weights[4*i :4*i + 4]))
        selection = []

        distributions = []
        for i in range(len(out)):
            if self.num_gaussians == 1:
                selection_weights = torch.ones(x.shape[0], 1)
            else:
                selection_weights = Sparsemax().forward(out[i][:, self.num_gaussians * 2 :])
            selection.append(selection_weights)
            mix = D.Categorical(selection_weights.cuda())
            comp = D.Independent(D.Normal(
                out[i][:, :self.num_gaussians].unsqueeze(2),
                out[i][:, self.num_gaussians:self.num_gaussians * 2].unsqueeze(2).exp() + 1e-7,), 1)
            gmm = MixtureSameFamily(mix, comp)
            distributions.append(gmm)
        return distributions, selection

    def validate(self, x, weights, lower, upper):
        samples, distributions = [], []
        means = []
        variances = []
        selection= []

        curr_input = x[:, 0].unsqueeze(1)
        for i in range(self.num_joints):
            out = self.joint_template(curr_input, weights[4*i: 4*i + 4])
            if self.num_gaussians == 1:
                selection_weights = torch.ones(x.shape[0], 1)
            else:
                #selection_weights = torch.softmax(out[:, self.num_gaussians * 2:], dim=1)
                selection_weights = Sparsemax().forward(out[:, self.num_gaussians*2 :])

            mix = D.Categorical(selection_weights.cuda())
            comp = D.Independent(D.Normal(
                out[:, :self.num_gaussians].unsqueeze(2),
                out[:, self.num_gaussians:self.num_gaussians * 2].unsqueeze(2).exp() + 1e-7, ), 1)

            means.append(out[:, :self.num_gaussians].unsqueeze(2))
            variances.append(out[:, self.num_gaussians:self.num_gaussians * 2].unsqueeze(2).exp())
            selection.append(selection_weights)

            dist = MixtureSameFamily(mix, comp)
            sample = dist.sample()
            sample = sample.clip(lower[i], upper[i])
            curr_input = torch.cat((curr_input,sample), dim = 1)
            samples.append(sample)
            distributions.append(dist)
        return samples, distributions, means, variances, selection
