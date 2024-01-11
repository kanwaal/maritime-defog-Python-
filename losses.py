# -*- coding: utf-8 -*-
"""Losses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yu6zxJ4Bum89DlgGzTa8rnZOohe4dco7

# Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TexturePriorLoss(nn.Module):
    def __init__(self, theta):
        super(TexturePriorLoss, self).__init__()
        self.theta = theta

    def forward(self, F, I):
        # Calculate gradients of F and I
        Fx = torch.abs(F[:, :, :, :-1] - F[:, :, :, 1:])  # Gradient of F in the x direction
        Fy = torch.abs(F[:, :, :-1, :] - F[:, :, 1:, :])  # Gradient of F in the y direction
        Ix = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])  # Gradient of I in the x direction
        Iy = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])  # Gradient of I in the y direction

        # Compute the texture prior loss
        Wx = 1 + torch.exp(-self.theta * Ix)  # Weight for x direction
        Wy = 1 + torch.exp(-self.theta * Iy)  # Weight for y direction

        # Calculate the loss for both directions
        loss_x = Wx * torch.abs(Fx - Ix)
        loss_y = Wy * torch.abs(Fy - Iy)

        # Combine losses from both directions
        loss = torch.mean(loss_x) + torch.mean(loss_y)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as Ff

class SmoothnessPriorLoss(nn.Module):
    def __init__(self):
        super(SmoothnessPriorLoss, self).__init__()

    def forward(self, G):
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        laplacian_kernel = laplacian_kernel.to(G.device)
        G_laplacian = Ff.conv2d(G, laplacian_kernel, padding=1)
        loss = torch.norm(G_laplacian, p='fro')  # Frobenius norm
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as Ff

def low_pass_filter(image, kernel_size=5):
    channels = image.size(1)
    padding = kernel_size // 2
    # Create a low-pass (average) filter kernel for each channel
    filter_kernel = torch.ones((channels, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)
    filter_kernel = filter_kernel.to(image.device)
    # Apply the low-pass filter to each channel
    filtered_image = Ff.conv2d(image, filter_kernel, padding=padding, groups=channels)
    return filtered_image


class StructurePriorLoss(nn.Module):
    def __init__(self):
        super(StructurePriorLoss, self).__init__()

    def forward(self, G, I):
        # Apply low-pass filter to the input image I
        I_s = low_pass_filter(I)
        # Calculate the loss as the Frobenius norm of the difference
        loss = torch.norm(G - I_s, p='fro')
        return loss

import torch
import torch.nn as nn

class ConstraintLoss(nn.Module):
    def __init__(self):
        super(ConstraintLoss, self).__init__()

    def forward(self, F, G, I):
        # Ensure F and G are non-negative and less than or equal to the intensity of I
        # This is equivalent to: 0 <= F <= I and 0 <= G <= I
        F_clamped = torch.clamp(F, min=0)
        G_clamped = torch.clamp(G, min=0)

        # Calculate the loss for F being greater than I and for F being negative
        F_loss = torch.mean((F_clamped - I).clamp(min=0) ** 2) + torch.mean((F - F_clamped) ** 2)

        # Ensure the glow layer G is colorless, i.e., the same across all color channels
        # Since G is grayscale but represented in RGB, all channels should be equal
        G_mean = torch.mean(G_clamped, dim=1, keepdim=True)
        G_color_loss = torch.mean((G_clamped - G_mean) ** 2)

        # Calculate the loss for G being greater than I (using G_mean since G is colorless) and for G being negative
        G_loss = torch.mean((G_mean - I).clamp(min=0) ** 2) + torch.mean((G - G_clamped) ** 2)

        # Combine losses
        total_loss = F_loss + G_loss + G_color_loss

        return total_loss