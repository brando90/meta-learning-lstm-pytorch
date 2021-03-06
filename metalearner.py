from __future__ import division, print_function, absolute_import

import pdb
import math
import torch
import torch.nn as nn


class MetaLSTMCell(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLSTMCell, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            hidden_size (int): should be 1
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = n_learner_params
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.cI = nn.Parameter(torch.Tensor(n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        # want initial forget value to be high and input value to be low so that 
        #  model starts with gradient descent
        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hx=None):
        """Args:
            inputs = [x_all, grad]:
                x_all (torch.Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (torch.Tensor of size [n_learner_params]): gradients from learner
            hx = [f_prev, i_prev, c_prev]:
                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                i (torch.Tensor of size [n_learner_params, 1]): input gate
                c (torch.Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        x_all, grad = inputs # x_all = lstm(grad_t, loss_t), grad i.e. x_all is the preprocessed 
        batch, _ = x_all.size()
        # hx i.e. previous forget, update & cell state from metalstm
        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.cI
            hx = [f_prev, i_prev, c_prev]
        # sort out inputs to gates and sort hidden state/memory from last metalstm
        f_prev, i_prev, c_prev = hx
        
        # f_t = sigmoid(W_f * [ lstm(grad_t, loss_t), theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [ lstm(grad_t, loss_t), theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params: theta^<t> = f^<t>*theta^<t-1> - i^<t>*grad^<t>
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad) # note, - sign is important cuz i_next is positive due to sigmoid activation

        return c_next.squeeze(), [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):

    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
            n_learner_params (int): number of learner's parameters
        """
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)

    def forward(self, inputs, hs=None):
        """Args:
            inputs = [loss, grad_prep, grad]
                loss (torch.Tensor of size [1, 2])
                grad_prep (torch.Tensor of size [n_learner_params, 2])
                grad (torch.Tensor of size [n_learner_params])

            hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
        """
        # sort out hidden states for lstm and meta-lstm
        if hs is None:
            hs = [None, None]
        # sort out input x^<t> to normal lstm
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep) # [1, 2] -> [n_learner_params, 2]
        xn_lstm = torch.cat((loss, grad_prep), 1) # [n_learner_params, 4]
        
        # normal lstm
        lstm_hxn = hs[0] # previous hx from normal lstm = (lstm_hn, lstm_cn)
        lstmhx, lstmcx = self.lstm(input=xn_lstm, hx=lstm_hxn)
        
        # optimizer lstm i.e. theta^<t> = f^<t>*theta^<t-1> + i^<t>*grad^<t>
        metalstm_hxn = hs[1] # previous hx from optimizer lstm = [metalstm_fn, metalstm_in, metalstm_cn]
        xn_metalstm = [lstmhx, grad] # note, the losses,grads are preprocessed by the lstm first before passing to metalstm [outputs_of_lstm, grad] = [ lstm(losses, grad_preps), grad]
        theta_next, metalstm_hs = self.metalstm(inputs=xn_metalstm, hx=metalstm_hxn)

        return theta_next, [(lstmhx, lstmcx), metalstm_hs]

