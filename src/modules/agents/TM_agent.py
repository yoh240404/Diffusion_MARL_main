import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
import time
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DDPM.ddpm import tools


class Teammate_model_agent(nn.Module):
    def __init__(self, input_shape, args): 
        super(Teammate_model_agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.Sample = tools(self.args)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        else:
            self.rnn = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.massage_size = self.args.n_agents * input_shape + (self.args.n_agents - 1) * args.rnn_hidden_dim 

        self.noise_size = (self.args.n_agents - 1) * (input_shape + args.rnn_hidden_dim)  


        self.v_embed = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.massage_size, args.attention_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.attention_hidden_size, args.n_actions)
        )
        self.q_embed = nn.Linear(self.massage_size, args.attention_hidden_dim) 
        self.k_embed = nn.Linear(args.rnn_hidden_dim, args.attention_hidden_dim) 

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_() 

    def forward(self, inputs, batch_size, hidden_state, teammate_model):  
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        t3 = time.time()
        predict_massage_list = self.Sample.sample(teammate_model, inputs, self.noise_size)
        t4 = time.time()
        pm = predict_massage_list[-1].view(batch_size * self.n_agents, -1)
        pm = pm[:, :self.massage_size]
        pm_repeat = pm.unsqueeze(dim=-2).repeat(1,self.n_agents,1).view(batch_size * self.n_agents * self.n_agents,-1)
        h_repeat = h.view(batch_size, self.n_agents, -1).repeat(1, self.n_agents, 1).view(batch_size * self.n_agents * self.n_agents, -1)
        value = self.v_embed(torch.cat([h_repeat, pm_repeat], dim=-1)).view(batch_size, self.n_agents, self.n_agents, self.n_actions)
        key = self.k_embed(h).unsqueeze(1)
        query = self.q_embed(pm_repeat).view(batch_size * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha_ = torch.bmm(key / (self.args.attention_hidden_dim ** (1/2)), query).view(batch_size, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha_[:, i, i] = -1e9
        alpha = F.softmax(alpha_, dim=-1).reshape(batch_size, self.n_agents, self.n_agents, 1)
        q_bias = alpha * value  
        q_0 = self.fc2(h)
        q = q_0 + 0.1 * torch.sum(q_bias, dim=1).view(batch_size*self.n_agents, self.n_actions)
        return q, h
