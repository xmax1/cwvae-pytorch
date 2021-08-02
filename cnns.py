import torch as tc
from torch import nn

from config import Config

leaky_relu = nn.LeakyReLU(negative_slope=0.2)  # TF default apparently


class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        # padding = VALID??
        tf1, tf2, tf3, tf4 = cfg.total_filters, cfg.total_filters*2, cfg.total_filters*4, cfg.total_filters*8 
        self.conv_net = nn.Sequential(nn.Conv2d(3, tf1, kernel_size=(4, 4), stride=(2, 2), padding='valid'), leaky_relu,
                                      nn.Conv2d(tf1, tf2, kernel_size=(4, 4), stride=(2, 2), padding='valid'), leaky_relu,
                                      nn.Conv2d(tf2, tf3, kernel_size=(4, 4), stride=(2, 2), padding='valid'), leaky_relu,
                                      nn.Conv2d(tf3, tf4, kernel_size=(4, 4), stride=(2, 2), padding='valid'), leaky_relu)

        # can be adapted to have a different hidden size
        test_out = self.conv_net(tc.rand((1, 3, 64, 64))).reshape(-1)
        nf = test_out.shape[-1]
        print('n features: ', nf)
        self.enc_dense_layers = {i:nn.Sequential(*list(nn.Sequential(nn.Linear(nf, nf), nn.ReLU())
                                                 for _ in range(1, cfg.enc_dense_layers)))
                                                 for i in range(1, cfg.levels)}

        self.tmp_abs_factor = cfg.tmp_abs_factor

    def forward(self, obs):

        # (m, t, c, d, d) (batch_size, ctx_len, channels, dim, dim)
        # ctx_len is dependent on the tmp_abs_factor and the level. Higher levels have smaller context len

        x = obs.reshape((-1,) + obs.shape[2:])
        x = self.conv_net(x)
        x = x.reshape(obs.shape[:2] + (-1,))

        layers = [x]
        for level in range(1, self.cfg.levels):
            layer = self.enc_dense_layers[level](x)
            
            timesteps_to_merge = self.tmp_abs_factor ** level  # number to merge increases with depth
            # Padding the time dimension.
            timesteps_to_pad = -layer.shape[1] % timesteps_to_merge  # the number needed to pad the timeaxis by to match the number of timesteps to merge
            layer = tc.pad(layer, ((0, 0), (0, timesteps_to_pad), (0, 0)))  # ((before, after), ... ), also pads with zeros by default
            # Reshaping and merging in time.
            layer = layer.reshape((layer.shape[0], -1, timesteps_to_merge, layer.shape[2]))
            layer = layers.sum(2)
            layers.append(layer)
            print(f"Input shape at level {level}: {layer.shape}")
        
        return layers


class Decoder(nn.Module):

    def __init__(self, cfg: Config):
        super(Decoder, self).__init__()

        emb_dim = 320
        self.dense1 = nn.Linear(emb_dim, cfg.channels_mult * 1024)  # channels_mult is 1 and this code only works if it is 1 not my fault

        tf1, tf2, tf3 = cfg.total_filters, cfg.total_filters * 2, cfg.total_filters * 4
        self.convt = nn.Sequential(nn.ConvTranspose2d(1024, tf1, kernel_size=5, stride=2, padding='valid'), leaky_relu,
                                   nn.ConvTranspose2d(tf1, tf2, kernel_size=5, stride=2, padding='valid'), leaky_relu,
                                   nn.ConvTranspose2d(tf2, tf3, kernel_size=6, stride=2, padding='valid'), leaky_relu,
                                   nn.ConvTranspose2d(tf2, tf3, kernel_size=6, stride=2, padding='valid'), nn.Tanh())

    
    def forward(self, bottom_layer_output):
        """
        Arguments:
            bottom_layer_output : Tensor
                State tensor of shape (batch_size, timesteps, feature_dim)

        Returns:
            Output video of shape (batch_size, timesteps, 64, 64, out_channels)
        """
        x = self.dense1(bottom_layer_output)
        x = x.reshape((-1, 1, 1, x.shape[-1]))  # (BxT, 1, 1, 1024)

        x = self.convt(x)

        return x.reshape(bottom_layer_output.shape[:2] + x.shape[1:])  # (B, T, 64, 64, C)




'''

class Decoder(nn.Module):
    """ States to Images Decoder."""
    c: Config

    @nn.compact
    def __call__(self, bottom_layer_output):
        """
        Arguments:
            bottom_layer_output : Tensor
                State tensor of shape (batch_size, timesteps, feature_dim)

        Returns:
            Output video of shape (batch_size, timesteps, 64, 64, out_channels)
        """
        x = nn.Dense(self.c.channels_mult * 1024)(bottom_layer_output)  # (B, T, 1024)
        # Merge batch and time dimensions, expand two (spatial) dims.
        x = jnp.reshape(x, (-1, 1, 1, x.shape[-1]))  # (BxT, 1, 1, 1024)

        ConvT = partial(nn.ConvTranspose, strides=(2, 2), padding='VALID')
        x = leaky_relu(ConvT(self.c.total_filters * 4, (5, 5))(x))  # (BxT, 5, 5, 128)
        x = leaky_relu(ConvT(self.c.total_filters * 2, (5, 5))(x))  # (BxT, 13, 13, 64)
        x = leaky_relu(ConvT(self.c.total_filters, (6, 6))(x))  # (BxT, 30, 30, 32)
        x = nn.tanh(ConvT(self.c.channels, (6, 6))(x))  # (BxT, 64, 64, C)
        return x.reshape(bottom_layer_output.shape[:2] + x.shape[1:])  # (B, T, 64, 64, C)




class Encoder_jax(nn.Module):
    """
    Multi-level Video Encoder.
    1. Extracts hierarchical features from a sequence of observations.
    2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
    3. Uses dense features for each level of the hierarchy above the bottom-most level.
    """
    c: Config

    @nn.compact
    def __call__(self, obs):
        """
        Arguments:
            obs : Tensor
                Un-flattened observations (videos) of shape (batch size, timesteps, height, width, channels)
        """
        # Merge batch and time dimensions.
        x = obs.reshape((-1,) + obs.shape[2:])

        Conv = partial(nn.Conv, kernel_size=(4, 4), strides=(2, 2), padding='VALID')
        x = leaky_relu(Conv(self.c.total_filters)(x))
        x = leaky_relu(Conv(self.c.total_filters * 2)(x))
        x = leaky_relu(Conv(self.c.total_filters * 4)(x))
        x = leaky_relu(Conv(self.c.total_filters * 8)(x))
        x = x.reshape(obs.shape[:2] + (-1,))
        layers = [x]
        print(f"Input shape at level 0: {x.shape}")

        feat_size = x.shape[-1]

        for level in range(1, self.c.levels):
            for _ in range(self.c.enc_dense_layers - 1):
                x = nn.relu(nn.Dense(self.c.enc_dense_embed_size)(x))
            if self.c.enc_dense_layers > 0:
                x = nn.Dense(feat_size)(x)
            layer = x
            timesteps_to_merge = self.c.tmp_abs_factor ** level  # number to merge increases with depth
            # Padding the time dimension.
            timesteps_to_pad = -layer.shape[1] % timesteps_to_merge  # the number needed to pad the timeaxis by to match the number of timesteps to merge
            layer = jnp.pad(layer, ((0, 0), (0, timesteps_to_pad), (0, 0)))  # ((before, after), ... ), also pads with zeros by default
            # Reshaping and merging in time.
            layer = layer.reshape((layer.shape[0], -1, timesteps_to_merge,
                                   layer.shape[2]))
            layer = jnp.sum(layer, axis=2)
            layers.append(layer)
            print(f"Input shape at level {level}: {layer.shape}")

        return layers

'''


if __name__ == '__main__':

    from types import SimpleNamespace

    cfg = SimpleNamespace(**dict(channels_mult=1, total_filters=10, levels=2, enc_dense_layers=2, tmp_abs_factor=3))
    print(cfg)
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)




    # unpacking test of sequential on sequential
    x = tuple(nn.Sequential(nn.Linear(5, 5), nn.ReLU()) for _ in range(1, 3))
    enc_dense_layers = {i:nn.Sequential(*list(nn.Sequential(nn.Linear(5, 5), nn.ReLU()) for _ in range(1, 2))) 
                                                 for i in range(1, 2)}

