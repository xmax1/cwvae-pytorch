from functools import partial
from itertools import islice

import torch as tc


import numpy as np
import wandb

from config import Config, parse_config
from cwvae import Model
from tools import log, video

if __name__ == "__main__":
    c = parse_config()
    with wandb.init(config=c):
        c = Config(**wandb.config)
        c.save()
        train_batches = c.load_dataset()
        val_batch = next(iter(c.load_dataset(eval=True)))
        model = Model(c)

        state = Adam(learning_rate=c.lr, eps=1e-4).create(params)


        def get_metrics(state, rng, obs):
            _, metrics = model.apply(state.target, obs=obs,
                                     rngs=dict(sample=rng))
            return metrics


        def get_video(state, rng, obs):
            return video(pred=model.apply(
                state.target, obs=obs, rngs=dict(sample=rng),
                method=model.open_loop_unroll), target=obs[:, c.open_loop_ctx:])

        
        
        if state.state.step:
            print(f"Restored model from {c.exp_rootdir}")
            print(f"Will start training from step {state.state.step}")
            train_batches = islice(train_batches, state.state.step, None)

        print("Training.")
        for train_batch in train_batches:
            step = state.step  # IMPLEMENT ME

            # Start 
            model.zero_grad()

            loss_fn, metrics = model(train_batch)

            loss = loss_fn(state.target)

            loss.backward()

            grads = 
            
            grad_norm = jnp.linalg.norm(jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
            
            if c.clip_grad_norm_by:
                # Clipping gradients by global norm
                scale = jnp.minimum(c.clip_grad_norm_by / grad_norm, 1)
                grad = jax.tree_map(lambda x: scale * x, grad)
            
            metrics['grad_norm'] = grad_norm
            
            state =  state.apply_gradient(grad)

            state, rng, metrics = train_step(state, rng, train_batch)

            # fin

            print(f"batch {step}: loss {metrics['loss']:.1f}")

            if state.step % c.save_scalars_every == 0:
                log(metrics, step, 'train/')
                log(get_metrics(state, rng, val_batch), step, 'val/')

            if c.save_gifs and step % c.save_gifs_every == 0:
                v = np.array(get_video(state, video_rng, train_batch))
                log(dict(pred_video=wandb.Video(v, fps=10)), step, 'train/')
                v = np.array(get_video(state, video_rng, val_batch))
                log(dict(pred_video=wandb.Video(v, fps=10)), step, 'val/')

            state.advance()

            # REPLACE SAVE MODEL

        print("Training complete.")
