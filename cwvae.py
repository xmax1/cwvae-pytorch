import torch as tc
from torch import nn
import torch.distributions as tfd

# from tensorflow_probability.substrates.jax import distributions as tfd

from cells import RSSMCell
from cnns import Encoder, Decoder
from config import Config



class CWVAE(nn.Module):


    def forward(self, data, use_observations=None, initial_state=None):
        """
        Used to unroll a list of recurrent cells.

        Arguments:
            data : list of encoded observations
                Number of timesteps at every level in 'inputs' is the number of steps to be unrolled.
            use_observations : None or list[bool]
            initial_state : list of cell states
        """
        if use_observations is None:
            use_observations = self.c.levels * [True]
        if initial_state is None:
            initial_state = self.c.levels * [None]

        cells = [RSSMCell(self.c) for _ in range(self.c.levels)]

        priors = []
        posteriors = []
        last_states = []
        top_level = self.c.levels - 1
        for level, (cell, use_obs, obs_inputs, initial) in \
                reversed(list(enumerate(zip(cells, use_observations, data, initial_state)))):

            print(f"Input shape in CWVAE level {level}: {obs_inputs.shape}")

            if level == top_level:  # is_top_level = True as top level is the max # level
                # Feeding in zeros as context to the top level:
                context = tc.zeros(obs_inputs.shape[:2] + (cell.state_size["output"],))
            else:
                # Tiling context from previous layer in time by tmp_abs_factor:
                context = tc.expand_dims(context, axis=2)
                context = tc.tile(context, [1, 1, self.c.tmp_abs_factor]
                                   + (len(context.shape) - 3) * [1])
                s = context.shape
                context = context.reshape((s[0], s[1] * s[2]) + s[3:])
                # Pruning timesteps to match inputs:
                context = context[:, :obs_inputs.shape[1]]

            # Unroll of RNN cell.
            call_cell = lambda state, xs: cell(state, xs, use_obs=use_obs)

            state = cell.zero_state(obs_inputs.shape[0]) if initial is None else initial

            # SCAN FUNCTION, DOES IT WORK?? NOBODY KNOWS
            split_contexts = context.split(1, dim=1, keepdims=False)
            split_obs_inputs = obs_inputs.split(1, dim=1, keepdims=False)

            priors = []
            posteriors = []    
            for c, o in zip(split_contexts, split_obs_inputs):
                state, (prior, posterior) = call_cell(state, (c, o))

                priors.append(prior)
                posteriors.append(posterior)

            prior = tc.stack(priors, dim=1)
            posterior = tc.stack(posteriors, dim=1)
        
            '''
            # call_cell = lambda state, xs: cell(state, xs, use_obs=use_obs)
            # xs = (obs_inputs, context)
            # shape of initial is: 
            # shape of obs_inputs is:
            # shape of context is: 
            scan = nn.scan(  # lifted version of jax.lax.scan
                call_cell,
                variable_broadcast='params',
                split_rngs=dict(params=False, sample=True),  # make_rng has been used inside the cells, if split the PNRG sequences are different in each iteration
                in_axes=1, out_axes=1)  # axis to scan over for arguments / return value 
            # does this mean scanning over the axis of the tensor or the argument number? in_axes usually has (n, m, t) tuple
            # sequence length goes along the first axis, it is how many we are rolling out by

            # initial is None or tensors, is it a list of length something? what is initial
            # context is a single tensor
            # obs_inputs is a single tensor
            last_state, (prior, posterior) = scan(cell, initial, (obs_inputs, context))
            '''

            context = posterior["output"]

            last_states.insert(0, state)  # puts it at the start of the list even though we are going in reverse
            priors.insert(0, prior)
            posteriors.insert(0, posterior)
        return last_states, priors, posteriors

    def open_loop_unroll(self, inputs):
        # what is in inputs? obs_encoded
        # is the maximum tmp_abs_factor ** level divisible by the open_loop_ctx
        assert self.c.open_loop_ctx % (self.c.tmp_abs_factor ** (self.c.levels - 1)) == 0, \
            f"Incompatible open-loop context length {self.open_loop_ctx} and " \
            f"temporal abstraction factor {self.tmp_abs_factor} for levels {self.levels}."
        ctx_lens = [self.c.open_loop_ctx // self.c.tmp_abs_factor ** level for level in range(self.c.levels)]
        # top level is higher number is the deeper hierarchy
        # top level has shorter ctx_len

        # 4 tmp_abs_factor 
        # 64 open_loop_ctx
        # level0 ctx_len = 64
        # level1 ctx_len = 16
        # level2 ctx_len = 8
        # level3 ctx_len = 1
        # in a line of 64 level0 is updated everytime, level1 is updated every 2, level2 is updated every 4, leaving totals 64, 32, 16

        # n_levels of inputs
        # splitting into pre and post inputs
        # pre_inputs are the inputs 
        # post_inputs are zeros
        pre_inputs, post_inputs = zip(*[(input[:, :ctx_len], tc.zeros_like(input[:, ctx_len:])) for input, ctx_len in zip(inputs, ctx_lens)])

        # this generates the last states??
        last_states, _, _ = self(pre_inputs, use_observations=self.c.use_observations)

        # this generates the new states?? 
        # are the inputs the observations... !!! they are the outputs of the encoder, which is a list of embedded features for each level
        _, predictions, _ = self(post_inputs, use_observations=self.c.levels * [False], initial_state=last_states)
        return predictions


class Model(nn.Module):
    c: Config

    def setup(self):
        self.encoder = Encoder(self.c)
        self.c.enc_dense_features = self.encoder.enc_dense_features
        self.model = CWVAE(self.c)
        self.decoder = Decoder(self.c)

    def decode(self, predictions):
        bottom_layer_output = predictions[0]['output']
        return self.decoder(bottom_layer_output)

    def forward(self, obs):
        assert obs.shape[-3:] == (self.c.channels, 64, 64)
        _, priors, posteriors = self.model(self.encoder(obs))
        output = tfd.Independent(tfd.Normal(self.decode(posteriors), self.c.dec_stddev))
        priors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"])) for d in priors]
        posteriors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"])) for d in posteriors]

        nll_term = -tc.mean(output.log_prob(obs), dim=0)
        kls = [tc.mean(posterior.kl_divergence(prior), dim=0) for prior, posterior in zip(priors, posteriors)]
        kl_term = sum(kls)
        metrics = dict(loss=nll_term + kl_term, kl_term=kl_term, nll_term=nll_term)
        for lvl, (prior, posterior, kl) in enumerate(zip(priors, posteriors, kls)):
            metrics.update({
                f"avg_kl_prior_posterior__level_{lvl}": kl,
                f"avg_entropy_prior__level_{lvl}": tc.mean(prior.entropy(), 0),
                f"avg_entropy_posterior__level_{lvl}": tc.mean(posterior.entropy(), 0)
            })

        per_timestep_metrics = {k: v / obs.shape[1] for k, v in metrics.items()}
        return per_timestep_metrics['loss'], per_timestep_metrics

    def open_loop_unroll(self, obs):
        return self.decode(self.model.open_loop_unroll(self.encoder(obs)))


'''

class CWVAE_jax(nn.Module):
    c: Config

    @nn.compact
    def __call__(self, inputs, use_observations=None, initial_state=None):
        """
        Used to unroll a list of recurrent cells.

        Arguments:
            inputs : list of encoded observations
                Number of timesteps at every level in 'inputs' is the number of steps to be unrolled.
            use_observations : None or list[bool]
            initial_state : list of cell states
        """
        if use_observations is None:
            use_observations = self.c.levels * [True]
        if initial_state is None:
            initial_state = self.c.levels * [None]

        cells = [RSSMCell(self.c) for _ in range(self.c.levels)]

        priors = []
        posteriors = []
        last_states = []
        is_top_level = True
        for level, (cell, use_obs, obs_inputs, initial) in reversed(list(
                enumerate(
                    zip(cells, use_observations, inputs, initial_state)))):

            print(f"Input shape in CWVAE level {level}: {obs_inputs.shape}")

            if is_top_level:
                # Feeding in zeros as context to the top level:
                context = jnp.zeros(obs_inputs.shape[:2] + (cell.state_size["output"],))
                is_top_level = False
            else:
                # Tiling context from previous layer in time by tmp_abs_factor:
                context = jnp.expand_dims(context, axis=2)
                context = jnp.tile(context, [1, 1, self.c.tmp_abs_factor]
                                   + (len(context.shape) - 3) * [1])
                s = context.shape
                context = context.reshape((s[0], s[1] * s[2]) + s[3:])
                # Pruning timesteps to match inputs:
                context = context[:, :obs_inputs.shape[1]]

            # Unroll of RNN cell.
            scan = nn.scan(
                lambda c, state, xs: c(state, xs, use_obs=use_obs),
                variable_broadcast='params',
                split_rngs=dict(params=False, sample=True),
                in_axes=1, out_axes=1)

            initial = cell.zero_state(obs_inputs.shape[0]
                                      ) if initial is None else initial
            last_state, (prior, posterior) = scan(cell, initial,
                                                  (obs_inputs, context))
            context = posterior["output"]

            last_states.insert(0, last_state)
            priors.insert(0, prior)
            posteriors.insert(0, posterior)
        return last_states, priors, posteriors

    def open_loop_unroll(self, inputs):
        assert self.c.open_loop_ctx % (self.c.tmp_abs_factor ** (self.c.levels - 1)) == 0, \
            f"Incompatible open-loop context length {self.open_loop_ctx} and " \
            f"temporal abstraction factor {self.tmp_abs_factor} for levels {self.levels}."
        ctx_lens = [self.c.open_loop_ctx // self.c.tmp_abs_factor ** level for level in range(self.c.levels)]
        # n_levels of inputs
        # splitting into pre and post inputs
        # pre_inputs are the inputs 
        # post_inputs are zeros
        pre_inputs, post_inputs = zip(*[(input[:, :ctx_len], jnp.zeros_like(input[:, ctx_len:])) for input, ctx_len in zip(inputs, ctx_lens)])

        # this generates the last states??
        last_states, _, _ = self(pre_inputs, use_observations=self.c.use_observations)

        # this generates the new states?? 
        # are the inputs the observations... !!! they are the outputs of the encoder, which is a list of embedded features for each level
        _, predictions, _ = self(post_inputs, use_observations=self.c.levels * [False], initial_state=last_states)
        return predictions


class Model(nn.Module):
    c: Config

    def setup(self):
        self.encoder = Encoder(self.c)
        self.model = CWVAE(self.c)
        self.decoder = Decoder(self.c)

    def decode(self, predictions):
        bottom_layer_output = predictions[0]['output']
        return self.decoder(bottom_layer_output)

    def __call__(self, obs):
        assert obs.shape[-3:] == (self.c.channels, 64, 64)
        _, priors, posteriors = self.model(self.encoder(obs))
        output = tfd.Independent(
            tfd.Normal(self.decode(posteriors), self.c.dec_stddev))
        priors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"]))
                  for d in priors]
        posteriors = [tfd.Independent(tfd.Normal(d["mean"], d["stddev"]))
                      for d in posteriors]

        nll_term = -jnp.mean(output.log_prob(obs), 0)
        kls = [jnp.mean(posterior.kl_divergence(prior), 0)
               for prior, posterior in zip(priors, posteriors)]
        kl_term = sum(kls)
        metrics = dict(loss=nll_term + kl_term,
                       kl_term=kl_term, nll_term=nll_term)
        for lvl, (prior, posterior, kl) in enumerate(
                zip(priors, posteriors, kls)):
            metrics.update({
                f"avg_kl_prior_posterior__level_{lvl}": kl,
                f"avg_entropy_prior__level_{lvl}": jnp.mean(prior.entropy(), 0),
                f"avg_entropy_posterior__level_{lvl}": jnp.mean(
                    posterior.entropy(), 0)
            })

        per_timestep_metrics = {k: v / obs.shape[1] for k, v in metrics.items()}
        return per_timestep_metrics['loss'], per_timestep_metrics

    def open_loop_unroll(self, obs):
        return self.decode(self.model.open_loop_unroll(self.encoder(obs)))

'''
if __name__ == '__main__':

    x = reversed(list(enumerate(range(10, 20))))
    print(x)

    open_loop_ctx = 36 
    tmp_abs_factor = 6
    levels = 3
    ctx_lens = [open_loop_ctx // tmp_abs_factor ** level for level in range(levels)]
    print(ctx_lens)
    # ctx_lens are then the time dimension of the network
