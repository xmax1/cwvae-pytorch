import torch as tc
from torch import nn

from config import Config


class RSSMPrior(nn.Module):

    def __init__(self, cfg: Config):
        super(RSSMPrior, self).__init__()
        self.cfg = cfg
        self.dense1 = nn.Linear(cfg.cell_stoch_size + (cfg.cell_stoch_size + cfg.cell_deter_size), cfg.cell_embed_size)  # cell embed size is the output size
        self.gru = nn.GRUCell(cfg.cell_embed_size, cfg.gru_hidden_prior)  # dims are the ins and outs of hl1, doesn't need the carry because the hidden state is maintained by pytorch
        self.dense2 = nn.Linear(cfg.gru_hidden_prior, cfg.cell_embed_size)
        self.dense3 = nn.Linear(cfg.cell_embed_size, cfg.cell_stoch_size)
        self.dense4 = nn.Linear(cfg.cell_embed_size, cfg.cell_stoch_size)

        self.relu = nn.ReLU(in_place=False)  # if true input is replaced by output in memory, uses less memory but only more efficient in some cases
        self.softplus = nn.Softplus(in_place=False)

    def forward(self, prev_state, context):
        '''
        prev_state: dict = {sample: tensor = (m, ?t, cell_stoch_size), ...
        context: tensor = (m, ?t, state_size['output'] = cell_stoch_size + cell_deter_size)
        '''
        data = tc.cat([prev_state['sample'], context], dim=-1)
        hl1 = self.relu(self.dense1(data))
        det_out = self.gru(hl1)
        hl2 = self.relu(self.dense2(det_out))
        mean = self.dense3(hl2)
        stddev = self.softplus(self.dense4(hl2 + 0.54)) + self.cfg.cell_min_stddev
        sample = tc.normal(mean, stddev)
        output = tc.cat([sample, det_out], dim=-1)
        return dict(mean=mean, stddev=stddev, sample=sample, det_out=det_out, output=output)


class RSSMPosterior(nn.Module):

    def __init__(self, cfg: Config):
        super(RSSMPosterior, self).__init__()
        self.cfg = cfg
        self.dense1 = nn.Linear(cfg.gru_hidden_prior + cfg.enc_dense_features, cfg.cell_embed_size)  # cell embed size is the output size
        self.dense2 = nn.Linear(cfg.cell_embed_size, cfg.cell_embed_size)
        self.dense3 = nn.Linear(cfg.cell_embed_size, cfg.cell_stoch_size)
        self.dense4 = nn.Linear(cfg.cell_embed_size, cfg.cell_stoch_size)

        self.relu = nn.ReLU(in_place=False)  # if true input is replaced by output in memory, uses less memory but only more efficient in some cases
        self.softplus = nn.Softplus(in_place=False)

    def forward(self, prior, obs_inputs):
        '''
        prior: dict = {det_out: tensor = (m, ?t, gru_hidden_prior)}
        '''
        data = tc.cat([prior['det_out'], obs_inputs], dim=-1)
        hl = self.relu(self.dense1(data))
        hl = self.relu(self.dense2(hl))
        mean = self.dense3(hl)
        stddev = self.softplus(self.dense4(hl))
        sample = tc.normal(mean, stddev)
        output = tc.cat([sample, prior['det_out']])
        return dict(mean=mean, stddev=stddev, sample=sample, det_out=prior['det_out'], output=output)


class RSSMCell(nn.Module):

    def __init__(self, cfg):
        super(RSSMCell, self).__init__()
        self.prior = RSSMPrior(cfg)
        self.posterior = RSSMPosterior(cfg)

    @property
    def state_size(self):
        return dict(
            mean=self.c.cell_stoch_size, stddev=self.c.cell_stoch_size,
            sample=self.c.cell_stoch_size, det_out=self.c.cell_deter_size,
            det_state=self.c.cell_deter_size,
            output=self.c.cell_stoch_size + self.c.cell_deter_size)

    def zero_state(self, batch_size, dtype=tc.float32):
        return {k: tc.zeros((batch_size, v), dtype=dtype)
                for k, v in self.state_size.items()}

    def forward(self, state, inputs, use_obs):
        context, obs_input = inputs
        '''
        state: dict = {sample: tensor = (m, ?t, cell_stoch_size), ...}
        context: tensor = (m, ?t, state_size['output'].shape[-1] = cell_stoch_size + cell_deter_size)
        obs_input: tensor = (m, ?t, enc_dense_features)
        '''
        
        prior = self.prior(state, context)
        posterior = self.posterior(prior, obs_input) if use_obs else prior
        return posterior, (prior, posterior)




'''
class RSSMCell_jax(nn.Module):
    c: Config

    @property
    def state_size(self):
        return dict(
            mean=self.c.cell_stoch_size, stddev=self.c.cell_stoch_size,
            sample=self.c.cell_stoch_size, det_out=self.c.cell_deter_size,
            det_state=self.c.cell_deter_size,
            output=self.c.cell_stoch_size + self.c.cell_deter_size)

    def zero_state(self, batch_size, dtype=jnp.float32):
        return {k: jnp.zeros((batch_size, v), dtype=dtype)
                for k, v in self.state_size.items()}

    @nn.compact
    def __call__(self, state, inputs, use_obs):
        obs_input, context = inputs
        prior = RSSMPrior(self.c)(state, context)
        posterior = RSSMPosterior(self.c)(prior,
                                          obs_input) if use_obs else prior
        return posterior, (prior, posterior)


class RSSMPosterior_jax(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, prior, obs_inputs):
        inputs = jnp.concatenate([prior["det_out"], obs_inputs], -1)
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(inputs))
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(hl))
        mean = nn.Dense(self.c.cell_stoch_size)(hl)
        stddev = nn.softplus(
            nn.Dense(self.c.cell_stoch_size)(hl + .54)) + self.c.cell_min_stddev
        dist = tfd.MultivariateNormalDiag(mean, stddev)
        sample = dist.sample(seed=self.make_rng('sample'))
        return dict(mean=mean, stddev=stddev, sample=sample,
                    det_out=prior["det_out"], det_state=prior["det_state"],
                    output=jnp.concatenate([sample, prior["det_out"]], -1))







class RSSMPrior_jax(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, prev_state, context):
        inputs = jnp.concatenate([prev_state["sample"], context], -1)
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(inputs))
        det_state, det_out = GRUCell()(prev_state["det_state"], hl)  # carry, inputs (hidden state, inputs) det state has been initialised at some point
        hl = nn.relu(nn.Dense(self.c.cell_embed_size)(det_out))
        mean = nn.Dense(self.c.cell_stoch_size)(hl)
        stddev = nn.softplus(
            nn.Dense(self.c.cell_stoch_size)(hl + .54)) + self.c.cell_min_stddev
        dist = tfd.MultivariateNormalDiag(mean, stddev)
        sample = dist.sample(seed=self.make_rng('sample'))
        return dict(mean=mean, stddev=stddev, sample=sample,
                    det_out=det_out, det_state=det_state,
                    output=jnp.concatenate([sample, det_out], -1))

'''