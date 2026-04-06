# Import necessary packages
import torch
import torch.nn as nn

class FourierL1Loss(nn.Module):
    def __init__(self):
        super(FourierL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, generated, gt):
        # compute FFT of both images
        gen_fft = torch.fft.fft2(generated)  # apply fourier transform
        gen_fft_shifted = torch.fft.fftshift(gen_fft)  # shift the zero frequency component to the centre

        gt_fft = torch.fft.fft2(gt)
        gt_fft_shifted = torch.fft.fftshift(gt_fft)

        # compute magnitude
        gen_mag = torch.abs(gen_fft_shifted)
        gt_mag = torch.abs(gt_fft_shifted)

        # compute l1 loss of the fourier images
        loss = self.l1_loss(gen_mag, gt_mag)
        return loss


BCE = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

def generator_loss(D: nn.Module, x, y, y_fake, alpha=0.1, sigma=100):
    # adversarial loss
    pred = D(x, y_fake)
    adv_loss = BCE(pred, torch.ones_like(pred))

    # l1 loss between the ground truth and generated image
    l1_loss = l1(y, y_fake)

    # l1 loss between fourier images
    Fy = torch.fft.fft2(y)
    Fy_fake = torch.fft.fft2(y_fake)

    Fy_shifted = torch.fft.fftshift(Fy)
    Fy_fake_shifted = torch.fft.fftshift(Fy_fake)

    abs_y = torch.abs(Fy_shifted)
    abs_y_fake = torch.abs(Fy_fake_shifted)

    fourier_l1_loss = l1(abs_y, abs_y_fake)

    # combine and return all losses as one value
    return adv_loss + (sigma * l1_loss) + (alpha * fourier_l1_loss)

# discriminator loss
def discriminator_loss(D: nn.Module, x, y, y_fake):
    real_pred = D(x, y)
    fake_pred = D(x, y_fake.detach())

    real_labels = torch.ones_like(real_pred)
    fake_labels = torch.zeros_like(fake_pred)

    loss_real = BCE(real_pred, real_labels)
    loss_fake = BCE(fake_pred, fake_labels)

    return 0.5 * (loss_real + loss_fake)