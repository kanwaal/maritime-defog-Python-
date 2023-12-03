import gc
import numpy as np
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

def is_grayscale(image):
    return len(image.shape) < 3 or (len(image.shape) == 3 and image.shape[2] == 1)
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


def decomposition(I, alpha, ii, beta, gamma):
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

    i = 0
    while True:
        i += 1
        prev_F = F.copy()
        lambda_ = min(2 ** (ii + i), 10 ** 5)
        Denormin_F = lambda_ * fft_double_grad + alpha * fft_double_laplace + beta
        qx = np.zeros_like(I)
        qy = np.zeros_like(I)
        qx = -convolve_rgb(F, f1) - Ix
        qy = -convolve_rgb(F, f2) - Iy
        qx = np.sign(qx) * np.maximum(np.abs(qx) - weight_x / lambda_, 0)
        qy = np.sign(qy) * np.maximum(np.abs(qy) - weight_y / lambda_, 0)
        Normin_q = np.concatenate((qx[:, -1:, :] - qx[:, :1, :], -np.diff(qx, 1, 1)), axis=1) + \
                   np.concatenate((qy[-1:, :, :] - qy[:1, :, :], -np.diff(qy, 1, 0)), axis=0)

        Normin_gN = fft_double_laplace * fft2(N.T).T

        FF = (lambda_ * (Normin_I + fft2(Normin_q.T).T) +
              alpha * (Normin_gI - Normin_gN) + beta * fft2((delta_I - N).T).T) / Denormin_F

        F = np.real(ifft2(FF.T).T)

        Normin_F = fft_double_laplace * fft2(F.T).T

        B = fft2((delta_I - F).T).T

        NN = (alpha * (Normin_gI - Normin_F) + beta * B) / Denormin_N

        N = np.real(ifft2(NN.T).T)

        print(np.sum(np.abs(prev_F - F)) / (H * W))
        if np.sum(np.abs(prev_F - F)) / (H * W) < 10 ** (-1):
            break
    for c in range(D):
        Ft = F[:, :, c]
        q = np.size(Ft)
        for k in range(500):
            m = np.sum(Ft[Ft < 0])
            n = np.sum(Ft[Ft > 1] - 1)
            dt = (m + n) / q
            if np.abs(dt) < 1 / q:
                break
            Ft = Ft - dt
        F[:, :, c] = Ft

    F = np.abs(F)
    F[F > 1] = 1

    N[N > 1] = 1
    N[N < 0] = 0
    N = np.mean(N, axis=2)

    G = np.abs(I - F - N[:, :, np.newaxis])
    G = np.min(G, axis=2)
    G = gaussian_filter(G, 3)

    F = np.abs(I - G[:, :, np.newaxis] - N[:, :, np.newaxis])
    F[F == 0] = 0.001

    return F, G, N

def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between two arrays of values.

    Parameters:
    y_true (numpy array or list): Actual values
    y_pred (numpy array or list): Predicted values

    Returns:
    float: The Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def show_images_side_by_side(img1, title1, img2, title2):
    # Load images
    # Check if images are grayscale
    is_img1_grayscale = is_grayscale(img1)
    is_img2_grayscale = is_grayscale(img2)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image with title
    if is_img1_grayscale:
        axes[0].imshow(img1, cmap='gray')
    else:
        axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')

    # Display the second image with title
    if is_img2_grayscale:
        axes[1].imshow(img2, cmap='gray')
    else:
        axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')

    # Show the figure
    plt.show()