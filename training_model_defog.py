# -*- coding: utf-8 -*-
"""training_model_defog.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yu6zxJ4Bum89DlgGzTa8rnZOohe4dco7
"""



import gc
import numpy as np
from itertools import product
import scipy.io
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage import convolve, gaussian_filter
from numpy.fft import fft2, ifft2, fftshift
import scipy.io
from scipy import signal
import cv2
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from PIL import Image
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import spdiags, eye
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2gray
from sklearn.neighbors import KDTree
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error as mse

def convolve_rgb(image, kernel):
    border_size = 1
    foggy_image_with_border = cv2.copyMakeBorder(image,
                                                 top=border_size,
                                                 bottom=border_size,
                                                 left=border_size,
                                                 right=border_size,
                                                 borderType=cv2.BORDER_WRAP)

    # Apply the filter
    temp = cv2.filter2D(foggy_image_with_border, -1, kernel)

    # Remove the added border
    temp = temp[border_size:-border_size, border_size:-border_size]

    return temp

def rgb2gray(rgb_image):
    """
    Convert an RGB image to a grayscale image.
    :param rgb_image: Input RGB image as a NumPy array.
    :return: Grayscale image as a NumPy array.
    """
    # Check if the input image is a 3D array (RGB image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be an RGB image")

    # Define the coefficients for the RGB channels
    coeffs = np.array([0.2989, 0.5870, 0.1140])

    # Convert the image to grayscale
    gray_image = np.dot(rgb_image[..., :3], coeffs)

    return gray_image

def convolve_rgb(image, kernel):
    border_size = 1
    foggy_image_with_border = cv2.copyMakeBorder(image,
                                                 top=border_size,
                                                 bottom=border_size,
                                                 left=border_size,
                                                 right=border_size,
                                                 borderType=cv2.BORDER_WRAP)

    # Apply the filter
    temp = cv2.filter2D(foggy_image_with_border, -1, kernel)

    # Remove the added border
    temp = temp[border_size:-border_size, border_size:-border_size]

    return temp

def gradient_weight(I):
    if len(I.shape) == 3 and I.shape[2] == 3:  # If the image has RGB channels
        I = np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale using standard weights

    lambda_val = 10
    f1 = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    f2 = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])

    Gx = signal.convolve2d(I, f1, mode='same')
    Gy = signal.convolve2d(I, f2, mode='same')

    ax = np.exp(-lambda_val * np.abs(Gx))
    thx = Gx < 0.01
    ax[thx] = 0
    weight_x = 1 + ax

    ay = np.exp(-lambda_val * np.abs(Gy))
    thy = Gy < 0.01
    ay[thy] = 0
    weight_y = 1 + ay

    return weight_x, weight_y


def psf2otf(psf, shape):
    psf = np.pad(psf, [(0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])], mode='constant')
    for axis, axis_size in enumerate(psf.shape):
        psf = np.roll(psf, -axis_size // 2, axis=axis)
    otf = fft2(psf.T).T
    return otf

def preprocess_for_training(foggy_image_path):
    # foggy_image_path = './Test_Data/V_08_01_0038.jpg' # This is relative path which needs to be updated.
    # mat_contents = scipy.io.loadmat('./mat_data_file/V_08_01_0038.mat')
    # ii = mat_contents['ii']
    # A = mat_contents['A']
    ii = 5

    # Read the image using OpenCV
    foggy_image = cv2.imread(foggy_image_path)

    # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
    foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)

    adjust_fog_removal = 2
    brightness = 2

    # Convert image from uint8 to float32 (similar to im2double)
    input_img = foggy_image.astype(np.float32) / foggy_image.max()
    I = input_img.copy()
    alpha = 20000
    beta = 0.1
    gamma = 10

    alpha = float(alpha)
    ii = float(ii)
    beta = float(beta)
    gamma = float(gamma)
    I = np.clip(I, 0, 1)
    gray = rgb2gray(I)
    H, W, D = I.shape

    weight_x, weight_y = gradient_weight(I)
    # f1 = np.array([[1, -1]])
    # f2 = np.array([[1], [-1]])
    f1 = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    f2 = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    f4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    gray = rgb2gray(I)
    I_filt = gaussian_filter(gray, sigma=10)
    delta_I = I - I_filt[..., np.newaxis]

    otfFx = psf2otf(f1, (H, W))
    otfFy = psf2otf(f2, (H, W))
    otfL = psf2otf(f4, (H, W))

    fft_double_laplace = np.abs(otfL) ** 2
    fft_double_grad = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        fft_double_grad = np.repeat(fft_double_grad[:, :, np.newaxis], D, axis=2)
        fft_double_laplace = np.repeat(fft_double_laplace[:, :, np.newaxis], D, axis=2)
        weight_x = np.repeat(weight_x[:, :, np.newaxis], D, axis=2)
        weight_y = np.repeat(weight_y[:, :, np.newaxis], D, axis=2)

    F = np.zeros_like(I)
    N = np.zeros_like(I)
    gray = rgb2gray(I)
    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)

    for channel in range(I.shape[2]):
        Ix[:, :, channel] = signal.convolve2d(I[:, :, channel], f1, mode='same')
        Iy[:, :, channel] = signal.convolve2d(I[:, :, channel], f2, mode='same')

    Normin_I = fft2((np.concatenate((Ix[:, -1:, :] - Ix[:, :1, :], -np.diff(Ix, 1, 1)), axis=1) +
                     np.concatenate((Iy[-1:, :, :] - Iy[:1, :, :], -np.diff(Iy, 1, 0)), axis=0)).T).T
    Denormin_N = gamma + alpha * fft_double_laplace + beta
    Normin_gI = fft_double_laplace * fft2(I.T).T

    i = 1
    prev_F = F.copy()
    lambda_ = min(2 ** (ii + i), 10 ** 5)
    Denormin_F = lambda_ * fft_double_grad + alpha * fft_double_laplace + beta

    # Update q
    qx = np.zeros_like(I)
    qy = np.zeros_like(I)
    qx = -convolve_rgb(F, f1) - Ix
    qy = -convolve_rgb(F, f2) - Iy
    qx = np.sign(qx) * np.maximum(np.abs(qx) - weight_x / lambda_, 0)
    qy = np.sign(qy) * np.maximum(np.abs(qy) - weight_y / lambda_, 0)
    # compute fog layer (F)
    Normin_q = np.concatenate((qx[:, -1:, :] - qx[:, :1, :], -np.diff(qx, 1, 1)), axis=1) + \
                np.concatenate((qy[-1:, :, :] - qy[:1, :, :], -np.diff(qy, 1, 0)), axis=0)

    Normin_gN = fft_double_laplace * fft2(N.T).T

    FF = (lambda_ * (Normin_I + fft2(Normin_q.T).T) +
          alpha * (Normin_gI - Normin_gN) + beta * fft2((delta_I - N).T).T) / Denormin_F

    F = np.real(ifft2(FF.T).T)

    # compute Noise
    Normin_F = fft_double_laplace * fft2(F.T).T

    B = fft2((delta_I - F).T).T

    NN = (alpha * (Normin_gI - Normin_F) + beta * B) / Denormin_N

    N = np.real(ifft2(NN.T).T)
    return(F)

from glob import glob
from tqdm import tqdm

paths = glob('/content/temp2/*.jpg')

input_images = []
for i in tqdm(range(len(paths))):
    path = paths[i]
    F = preprocess_for_training(path)
    input_images.append(F)

import numpy as np
import torch.optim as optim
import torch

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3,padding='same')
        self.conv2 = nn.Conv2d(6, 3, 3,padding='same')
        self.conv3 = nn.Conv2d(3, 6, 3,padding='same')
        self.conv4 = nn.Conv2d(6, 1, 3,padding='same')

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.sigmoid(self.conv2(x1))
        x2 = F.relu(self.conv3(x))
        x2 = F.sigmoid(self.conv4(x2))
        return x1,x2


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criteria= torch.nn.MSELoss()

"""# Losses"""

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

theta = 0.8
texture_loss_fn = TexturePriorLoss(theta)
SmoothnessPriorLoss_fn = SmoothnessPriorLoss()
StructurePriorLoss_fn = StructurePriorLoss()
constraint_loss_fn = ConstraintLoss()

for i in range(10):
    for j in range(len(input_images)):
        optimizer.zero_grad()
        inputs = input_images[j].copy()
        inputs = np.array([np.moveaxis(inputs,-1,0)])
        inputs = torch.from_numpy(inputs).float()
        outputs1,outputs2 = net(inputs)

        texture_loss = texture_loss_fn(outputs1, inputs)
        SmoothnessPrior_loss = SmoothnessPriorLoss_fn(outputs2)
        StructurePrior_loss = StructurePriorLoss_fn(outputs2, inputs)
        constraint_loss_value = constraint_loss_fn(outputs1, outputs2, inputs)
        loss = constraint_loss_value + texture_loss + SmoothnessPrior_loss + StructurePrior_loss
        loss.backward()
        optimizer.step()

model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

model = torch.jit.load('model_scripted.pt')
model.eval()