from functools import partial
from itertools import islice
import sys
sys.path.append(r'C:\Users\amwil\OneDrive\projects\msm')

import torch as tc
from torch.optim import Adam

import numpy as np
import wandb

from config import Config, parse_config
from cwvae import Model
from tools import video


def compute_grad_norm(model):
    grads = []
    for k, p in model.parameters():
        g = p.grad
        grads.append(tc.linalg.norm(g, dim=-1))
    return tc.linalg.norm(grads)


def map_grads(model, function, **args):
    for k, p in model.parameters():
        g = p.grad
        g = function(g, **args)
        p.grad = g


def scale_grads(grad, scale):
    return grad * scale


if __name__ == "__main__":
    c = parse_config()
    # with wandb.init(config=c):
    # c = Config(**wandb.config)
    # c = Config(**c)
    c.save()
    train_batches = c.load_dataset()
    val_batch = next(iter(c.load_dataset(eval=True)))
    model = Model(c)

    opt = Adam(model.parameters(), lr=0.001)

    def get_metrics(state, rng, obs):
        _, metrics = model.apply(state.target, obs=obs,
                                    rngs=dict(sample=rng))
        return metrics


    def get_video(state, rng, obs):
        return video(pred=model.apply(
            state.target, obs=obs, rngs=dict(sample=rng),
            method=model.open_loop_unroll), target=obs[:, c.open_loop_ctx:])


    print("Training.")
    for step, train_batch in enumerate(train_batches):

        # Start 
        model.zero_grad()

        loss, metrics = model(train_batch)

        loss.backward()

        grad_norm = compute_grad_norm(model)
        # tree_map maps a function over the leaves of the tree (all the grads)
        # tree_leaves recovers a list of all the leaves
        # how tf is norm applied to that?? 
        # grad_norm = jnp.linalg.norm(jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
        
        if c.clip_grad_norm_by:  # any non-zero float returns true
            # Clipping gradients by global norm
            scale = tc.minimum(c.clip_grad_norm_by / grad_norm, 1.)
            map_grads(model, scale_grads, scale)
            # grad = jax.tree_map(lambda x: scale * x, grad)
        
        metrics['grad_norm'] = grad_norm
        
        opt.step()

        # fin
        print(f"batch {step}: loss {metrics['loss']:.1f}")

        # REPLACE SAVE MODEL

    print("Training complete.")
