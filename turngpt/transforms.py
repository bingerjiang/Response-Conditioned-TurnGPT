import math
import torch
import torch.nn as nn
import numpy as np

from turngpt.turngpt_utils import get_positive_and_negative_indices


class ClassificationLabelTransform(object):
    def __init__(
        self,
        ratio=1,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        max_batch_size=256,
        unigram=True,
    ):
        self.ratio = ratio
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.unigram = unigram

        self.max_batch_size = max_batch_size

    def _unigram(self, x, input_ids):
        pos, neg = get_positive_and_negative_indices(
            input_ids, self.sp1_idx, self.sp2_idx, self.pad_idx
        )

        # positive
        n_pos = len(pos[0])
        pos_x = x[pos]

        # negative
        n_neg = len(neg[0])
        N = int(n_pos / self.ratio)
        neg_idx = torch.from_numpy(np.random.choice(n_neg, N)).long()
        neg_x = x[neg][neg_idx]

        pos_inp = input_ids[pos]
        neg_inp = input_ids[neg][neg_idx]
        inp = torch.cat((pos_inp, neg_inp))

        x = torch.cat((pos_x, neg_x))
        y = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        y[:n_pos] = 1
        return x, y, inp

    def onehot_speaker_shift(self, x, input_ids):
        sp = input_ids == self.sp1_idx
        sp += input_ids == self.sp2_idx
        sp = torch.cat(
            (sp[:, 1:], torch.zeros(sp.shape[0], 1).to(input_ids.device)), dim=-1
        )
        return x, sp, input_ids

    def __call__(self, x, input_ids):

        if self.unigram:
            return self._unigram(x, input_ids)
        else:
            return self.onehot_speaker_shift(x, input_ids)


class Gaussian1D(nn.Module):
    """
    Inspiration:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/2
    """

    def __init__(self, channels=1, kernel_size=5, sigma=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.gaussian_filter = self._gaussian_kernel(
            self.kernel_size, self.sigma, self.channels
        )

    def _gaussian_kernel(self, kernel_size=3, sigma=2, channels=1):
        """
        From:
            https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        """

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size).unsqueeze(-1)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((x_coord - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # reshape for nn.Conv1d
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
        # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)

        gaussian_filter = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        return self.gaussian_filter(x)


if __name__ == "__main__":

    from argparse import ArgumentParser
    from turngpt.acousticDM import AudioDM
    from ttd.tokenizer_helpers import convert_ids_to_tokens
    from turngpt.turngpt_utils import get_speaker_shift_indices
    from turngpt.models import gradient_check_batch, gradient_check_word_time
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        f0=True,
        waveform=True,
        f0_normalize=True,
        f0_interpolate=True,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.batch_size = 16
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()

    batch = next(iter(loader))

    batch_transform = ClassificationLabelTransform(
        ratio=1.0,
        sp1_idx=dm.sp1_idx,
        sp2_idx=dm.sp2_idx,
        pad_idx=dm.pad_idx,
        unigram=False,
    )

    input_ids = batch["input_ids"]
    x = torch.stack((batch["f0"], batch["rms"]), dim=-1)
    print("pros: ", x.shape)
    x, y, inp = batch_transform(x, input_ids)
    print("x: ", x.shape, x.device, x.dtype)
    print("y: ", y.shape, y.device, y.dtype)

    tokens = convert_ids_to_tokens(inp, dm.tokenizer)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(x)):
        ax.cla()
        ax.set_title(f"{tokens[i]}, label: {y[i].item()}")
        ax.plot(x[i, :, 0])
        plt.pause(0.01)
        input()
