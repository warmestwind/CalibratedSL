import torch.nn as nn
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange
from Cross_module import CrossAttention


num_pt = 4

def make_model(args):
    return Unet(args)

class Unet(nn.Module):
    def __init__(self, config):
        super(Unet, self).__init__()
        filter_config = (32, 64, 128, 256)
        self.depth = len(filter_config)
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        self.out_channels = num_pt + 1

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()


        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        self.bottom_conv = nn.Sequential(*[nn.Conv2d(256, 512, 3, 1, 1),
                           nn.BatchNorm2d(512),
                           nn.ReLU(), nn.Conv2d(512, 256, 3, 1, 1),
                           nn.BatchNorm2d(256),
                           nn.ReLU()])

        for i in range(0, self.depth):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                               decoder_filter_config[i + 1],
                                               decoder_n_layers[i]))


        # final classifier (equivalent to a fully connected layer)
        self.classifier_evi = nn.Conv2d(filter_config[0], filter_config[0], 3, 1, 1)
        self.classifier_evi1 = nn.Conv2d(filter_config[0], filter_config[0], 3, 1, 1)
        self.classifier_evi2 = nn.Conv2d(filter_config[1], filter_config[0], 3, 1, 1)
        self.classifier_evi3 = nn.Conv2d(filter_config[2], filter_config[0], 3, 1, 1)
        self.cls_evi = nn.ModuleList([self.classifier_evi3, self.classifier_evi2, self.classifier_evi1, self.classifier_evi])


        att_in = 16
        self.catt = CrossAttention(dim=att_in, heads=1, dim_head=32)
        self.num_pro =(num_pt+1)*2
        self.patch_size = 2
        self.to_patch_embedding_small = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.num_pro*self.patch_size*self.patch_size, att_in),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, int(256*256/self.patch_size/self.patch_size), att_in))

        self.cali_conv2 = nn.Conv2d(att_in, self.out_channels, 3, 1, 1)
        self.cali_conv3 = nn.AdaptiveAvgPool2d(1)

        # Pro
        self.input_dim = filter_config[0]
        self.Pro = nn.Parameter(torch.Tensor(1, self.num_pro, self.input_dim, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor( self.num_pro, 1))
        self.alpha = nn.Parameter(torch.Tensor( self.num_pro, 1))
        self.out_evi = nn.Conv2d(att_in, self.out_channels, 1, 1)
        self.classifier_u = nn.Conv2d(filter_config[0],  self.num_pro, 3, 1, 1)

        self.eps = 1

        self._init_weights()
        nn.init.normal_(self.Pro)
        nn.init.constant_(self.gamma, 1)
        nn.init.constant_(self.alpha, 1)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x
        feat_encoders = []
        # encoder path, keep track of pooling indices and features size
        for i in range(0, self.depth):
            feat_ori, (feat, ind), size = self.encoders[i](feat)
            feat_encoders.append(feat_ori)
            indices.append(ind)
            unpool_sizes.append(size)

        feat = self.bottom_conv(feat)
        feat_dec = feat


        # decoder path, upsampling with corresponding indices and size
        S_all = []
        for i in range(0, self.depth):
            feat_dec = self.decoders[i](feat_dec, feat_encoders[self.depth-i-1], indices[self.depth -1 - i], unpool_sizes[self.depth -1 - i])

            evi_f = self.cls_evi[i](feat_dec)

            [batch_size, _, height, weight] = feat_dec.size()
            s = torch.zeros(batch_size, self.num_pro, height, weight, device=feat_dec.device)
            # evi_map -> pro
            for k in range(self.num_pro):
                d = (evi_f - self.Pro[:, k])
                d = 0.5 * (d ** 2).sum(1)
                d = torch.exp(-self.gamma[k] ** 2 * d)
                d = self.alpha[k] * d
                s[:, k] = d
            S_all.append(F.interpolate(s, scale_factor=2**(self.depth-1-i), mode='bilinear'))

        s = torch.stack(S_all).sum(0)

        u_f = self.classifier_u(feat_dec)

        # cross att
        evi_f = self.to_patch_embedding_small(s)  # b, n=h*w, 128
        u_f = self.to_patch_embedding_small(u_f)

        # add pos
        evi_f += self.pos_embedding
        u_f += self.pos_embedding

        evi_f, u_f = self.catt(evi_f, u_f)

        if self.patch_size != 1:
            evi_f = F.interpolate(evi_f, scale_factor=self.patch_size, mode='bilinear')
            u_f = F.interpolate(u_f, scale_factor=self.patch_size, mode='bilinear')


        v = self.cali_conv2(u_f)
        v = F.softplus(self.cali_conv3(v))

        e = F.softplus(self.out_evi(evi_f))


        alpha = e + 1

        uncertainty = (v + self.eps) / (torch.sum(e+v+self.eps, dim=1, keepdim=True))  # num_pt+1 / S



        results = {'e': e, 'v': v, 'u': uncertainty, 'alpha': alpha}


        return results



class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return output, F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(n_in_feat, n_in_feat, 3, 2, 1, 1)

        layers = [nn.Conv2d(2*n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU()]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU()]

        self.features = nn.Sequential(*layers)

    def forward(self, x, x_e, indices, size):
        x = torch.cat([x_e, self.up_conv(x)], 1)
        return self.features(x)


if __name__ == '__main__':
    from config import get_config
    BUNET = Unet(get_config())


    BUNET(torch.ones(2, 1, 256, 256))


