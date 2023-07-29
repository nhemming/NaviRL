"""
Builds networks with multiple heads if needed.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self,device, input_dict, network_dict, output_dim):
        nn.Module.__init__(self)

        self.device = device

        head_sizes = OrderedDict()
        for name, value in input_dict.items():
            head_sizes[name] = len(value)

        self.network_dict = network_dict

        # create layers
        self.head_idx, self.tail_idx, self.layers = self.create_layers(head_sizes, network_dict, output_dim)

        # create the heads of the network
        #self.head_layers, concat_size = self.create_heads(head_sizes, network_dict)

        # create the tail of the network
        #self.tail_layers = self.create_tail(concat_size, network_dict, output_dim)

        # create dropout operators
        self.head_dropout, self.tail_dropout = self.create_dropout()

        # initialize the network
        self.initialize_layers(network_dict)

    def get_activation(self, key):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": nn.ELU(), "hardshrink": nn.Hardshrink(), "hardtanh": nn.Hardtanh(),
                                        "leakyrelu": nn.LeakyReLU(), "logsigmoid": nn.LogSigmoid(), "prelu": nn.PReLU(),
                                        "relu": nn.ReLU(), "relu6": nn.ReLU6(), "rrelu": nn.RReLU(), "selu": nn.SELU(),
                                        "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(),
                                        "logsoftmax": nn.LogSoftmax(),
                                        "softshrink": nn.Softshrink(), "softsign": nn.Softsign(), "tanh": nn.Tanh(),
                                        "tanhshrink": nn.Tanhshrink(), "softmin": nn.Softmin(),
                                        "softmax": nn.Softmax(dim=1),
                                        "none": None}
        return str_to_activations_converter[key]

    def get_initializer(self, key):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": nn.init.uniform_, "normal": nn.init.normal_,
                                        "eye": nn.init.eye_,
                                        "xavier_uniform": nn.init.xavier_uniform_, "xavier": nn.init.xavier_uniform_,
                                        "xavier_normal": nn.init.xavier_normal_,
                                        "kaiming_uniform": nn.init.kaiming_uniform_,
                                        "kaiming": nn.init.kaiming_uniform_,
                                        "kaiming_normal": nn.init.kaiming_normal_, "he": nn.init.kaiming_normal_,
                                        "orthogonal": nn.init.orthogonal_, "default": "use_default"}
        return str_to_initialiser_converter[key]

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return nn.Dropout(p=self.dropout)

    def create_layers(self,  head_sizes, network_dict, output_dim):
        layers = nn.ModuleList([])
        head_idx = OrderedDict()
        concat_size = 0
        k = 0
        for name, value in head_sizes.items():
            head_idx[k] = name
            input_dim = value
            tmp_layer_desc = network_dict[name]
            if isinstance(tmp_layer_desc['hidden_layers'], int):
                node_arr = [int(tmp_layer_desc['hidden_layers'])]
            else:
                node_arr = [int(i) for i in tmp_layer_desc['hidden_layers'].split(',')]

            for num_nodes in node_arr:
                # device
                layers.extend([nn.Linear(input_dim, num_nodes).to(self.device)])
                input_dim = num_nodes

                k += 1
            concat_size += input_dim

        tail_idx = k
        input_dim = concat_size
        nodes = network_dict['tail']['hidden_layers']
        if isinstance(nodes, int):
            node_arr = [int(nodes)]
        else:
            node_arr = [int(i) for i in nodes.split(',')]
        for num_nodes in node_arr:
            layers.extend([nn.Linear(input_dim, num_nodes).to(self.device)])
            input_dim = num_nodes
        layers.extend([nn.Linear(input_dim, output_dim).to(self.device)])

        return head_idx, tail_idx, layers

    def create_heads(self, head_sizes, network_dict):
        head_layers = OrderedDict()
        concat_size = 0
        for name, value in head_sizes.items():
            linear_layers = nn.ModuleList([])
            input_dim = value
            tmp_layer_desc = network_dict[name]
            if isinstance(tmp_layer_desc['hidden_layers'],int):
                node_arr = [int(tmp_layer_desc['hidden_layers'])]
            else:
                node_arr = [int(i) for i in tmp_layer_desc['hidden_layers'].split(',')]

            for num_nodes in node_arr:
                # device
                linear_layers.extend([nn.Linear(input_dim, num_nodes).to(self.device)])
                input_dim = num_nodes

            head_layers[name] = linear_layers

            concat_size += input_dim

        return head_layers, concat_size

    def create_tail(self, concat_size, network_dict, output_dim):
        tail_layers = nn.ModuleList([])
        input_dim = concat_size
        nodes = network_dict['tail']['hidden_layers']
        if isinstance(nodes,int):
            node_arr = [int(nodes)]
        else:
            node_arr = [int(i) for i in nodes.split(',')]
        for num_nodes in node_arr:
            tail_layers.extend([nn.Linear(input_dim, num_nodes).to(self.device)])
            input_dim = num_nodes
        tail_layers.extend([nn.Linear(input_dim, output_dim).to(self.device)])
        return tail_layers

    def create_dropout(self):
        head_dropout = OrderedDict()
        tail_dropout = None
        for name, value in self.network_dict.items():
            if 'head' in name:
                head_dropout[name] = nn.Dropout(p=value['dropout'])
            elif 'tail' in name:
                tail_dropout = nn.Dropout(p=value['dropout'])
        return head_dropout, tail_dropout

    def initialize_layers(self, network_dict):
        initializer = self.get_initializer(network_dict['initializer'])

        # initialize the layers
        for lay in self.layers:
            initializer(lay.weight)

    def forward(self, input_data):

        # pass through heads
        head_out = torch.FloatTensor().to(self.device)
        k = 0
        head_name = ''
        x = None
        for i, layer in enumerate(self.layers):
            if i == k and k in list(self.head_idx.keys()): # self.head_idx[k]:
                if i != 0:
                    head_out = torch.cat([head_out, x])
                head_name = self.head_idx[k]
                k += 1
                active_name = self.network_dict[head_name]['hidden_activation'] # TODO get activation for head
                x = input_data[head_name]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float().to(self.device)

            elif i == self.tail_idx:
                # complete the concatination
                x = torch.cat([head_out, x])
                active_name = self.network_dict['tail']['hidden_activation']

            if i == len(self.layers) - 1:
                x = layer(x)
                active_name = self.network_dict['tail']['last_activation']
                if active_name != 'none':
                    x = self.get_activation(active_name)(x)
            else:
                x = self.get_activation(active_name)(layer(x))
            # if self.batch_norm:
            #    x = self.batch_norm_layers[idx](x)
            if self.network_dict[head_name]['dropout'] != 0.0:
                x = self.head_dropout[head_name](x)

        return x