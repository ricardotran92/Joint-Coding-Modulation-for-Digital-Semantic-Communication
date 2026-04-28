from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import init
from modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize, ResidualBlock


def canonical_mod_method(mod_method):
    """Convert modulation method to a canonical form for consistent processing."""
    method = str(mod_method).strip().lower()
    # Accept accidentally double-quoted values from notebooks, e.g. "'64qam'" or "\"64qam\"".
    while len(method) >= 2 and (
        (method[0] == "'" and method[-1] == "'") or
        (method[0] == '"' and method[-1] == '"')
    ):
        method = method[1:-1].strip().lower()
    aliases = {
        "qpsk": "4qam",
        "4-qam": "4qam",
        "16-qam": "16qam",
        "64-qam": "64qam",
    }
    return aliases.get(method, method)


def modulation(logits, device, mod_method='bpsk'):
    mod_method = canonical_mod_method(mod_method)

    discrete_code = gumbel_softmax(logits, hard=True, tau=1.5)

    if mod_method == 'bpsk':
        output = discrete_code[:, :, 0] * (-1) + discrete_code[:, :, 1] * 1

    elif mod_method == '4qam':
        const = [1, -1]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    elif mod_method == '16qam':
        const = [-3, -1, 1, 3]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    elif mod_method == '64qam':
        const = [-7, -5, -3, -1, 1, 3, 5, 7]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    else:
        print("Modulation method not defined.")

    return output


class JCM(nn.Module):
    def __init__(self, config, device):
        super(JCM, self).__init__()
        self.config = config
        self.device = device
        self.mod_method = canonical_mod_method(self.config.mod_method)

        # define the number of probability categories
        if self.mod_method == 'bpsk':
            self.num_category = 2
        elif self.mod_method == '4qam':
            self.num_category = 2
        elif self.mod_method == '16qam':
            self.num_category = 4
        elif self.mod_method == '64qam':
            self.num_category = 8
        else:
            raise ValueError(
                f"Unsupported mod_method: {self.config.mod_method}. "
                "Use one of: bpsk, 4qam, 16qam, 64qam."
            )

        self.encoder = Encoder(self.config)

        if self.mod_method == 'bpsk':
            self.prob_convs = nn.Sequential(
                nn.Linear(config.channel_use * 4 * 4, config.channel_use * self.num_category),
                nn.ReLU(),
            )
        else:
            self.prob_convs = nn.Sequential(
                nn.Linear(config.channel_use * 2 * 4 * 4, config.channel_use * 2 * self.num_category),
                nn.ReLU(),
            )

        self.decoder_recon = Decoder_Recon(self.config)

        if self.mod_method == 'bpsk':
            self.decoder_class = Decoder_Class(int(config.channel_use / 2), int(config.channel_use / 8))
        else:
            self.decoder_class = Decoder_Class(int(config.channel_use * 2 / 2), int(config.channel_use * 2 / 8))

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)

    def reparameterize(self, probs):
        code = modulation(probs, self.device, self.mod_method)
        return code

    def forward(self, x):
        x_f = self.encoder(x).reshape(x.shape[0], -1)
        z = self.prob_convs(x_f).reshape(x.shape[0], -1, self.num_category)
        code = self.reparameterize(z)

        power, z = normalize(code)

        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train, z, self.device)
        if self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test, z, self.device)

        recon = self.decoder_recon(z_hat)
        r_class = self.decoder_class(z_hat)

        return code, z, z_hat, r_class, recon




