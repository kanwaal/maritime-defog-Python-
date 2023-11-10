import numpy as np
import scipy.io
import skimage.filters
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import scipy.io
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage import convolve, gaussian_filter

import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage import convolve, gaussian_filter
from scipy.fftpack import fft2, ifft2, fftshift


def lemire_nd_maxengine(A, idx, window, shapeflag):
    original_A_dim = len(A.shape)  # Store the original dimension of A

    # Inputs validation
    if A.shape == idx.shape:
        print('The dimensions of the two arrays are the same.')

    sz = A.shape
    if len(sz) == 2:
        A = A[:, :, np.newaxis]  # Adding a new third dimension
        sz = A.shape

    if len(sz) == 3 and len(idx.shape) == 2:
        if sz[0] == 1:  # Special case where first dimension of A is 1
            idx = np.reshape(idx, (1, idx.shape[0], idx.shape[1]))
        else:
            idx = np.reshape(idx, (idx.shape[0], idx.shape[1], 1))

    if len(A.shape) != 3 or len(idx.shape) != 3:
        raise ValueError('Not enough input arguments.')


    p, n, q = A.shape
    pi, ni, qi = idx.shape

    if pi != p or ni != n:
        raise ValueError('A and idx must have the same first two dimensions.')
    if qi == 0:
        qi = 1
    if q != qi:
        raise ValueError('A and idx must have the same third dimension or idx should not have a third dimension.')

    # Initialize outputs
    maxval = np.zeros(A.shape)
    maxidx = np.zeros(idx.shape)

    for j in range(q):
        for k in range(p):
            a = A[k, :, j]
            current_idx = idx[k, :, min(j, qi-1)]
            nWedge = 0
            Wedgefirst = 0
            Wedgelast = -1
            left = -window
            Wedge = np.zeros(n, dtype=int)  # Explicitly set dtype to int

            # Loop over the second dimension
            for i in range(n):
                left += 1

                # Update the wedge
                while nWedge > 0 and a[Wedge[Wedgelast]] <= a[i]:
                    nWedge -= 1
                    Wedgelast -= 1
                if nWedge > 0 and Wedge[Wedgefirst] <= left - window:
                    nWedge -= 1
                    Wedgefirst += 1
                nWedge += 1
                Wedgelast += 1
                Wedge[Wedgelast] = int(i)  # Explicitly cast to int

                # Retrieve the max value and its index
                if i >= window:
                    maxval[k, i-window, j] = a[Wedge[Wedgefirst]]
                    maxidx[k, i-window, j] = current_idx[Wedge[Wedgefirst]]

    # Handle the shapeflag
    if shapeflag == 1: # valid
        maxval = maxval[:, window-1:, :]
        maxidx = maxidx[:, window-1:, :]
    elif shapeflag == 3: # full
        maxval = np.vstack([np.zeros((p, window-1, q)), maxval])
        maxidx = np.vstack([np.zeros((p, window-1, q)), maxidx])

    if original_A_dim == 2:
        maxval = maxval.squeeze(axis=2)
        maxidx = maxidx.squeeze(axis=2)

    return maxval, maxidx


def lemire_nd_minengine(A, idx, window, shapeflag):
    if A.shape != idx.shape:
        raise ValueError("A and idx must have the same shape.")

    if window < 1:
        raise ValueError("window must be 1 or greater.")

    p, n, q = A.shape[0], A.shape[1], A.shape[2]
    margin = window - 1
    size = window + 1
    Wedge = np.zeros(size, dtype=int)

    if shapeflag == 3:  # FULL_SHAPE
        lstart = -margin
        dimOut1 = n + margin
    elif shapeflag == 2:  # SAME_SHAPEx
        lstart = -margin // 2
        dimOut1 = n
    else:  # VALID_SHAPE
        lstart = 0
        dimOut1 = n - margin

    imax = dimOut1 + margin + lstart

    minval = np.empty((p, dimOut1, q), dtype=A.dtype)
    minidx = np.empty((p, dimOut1, q))

    for j in range(q):
        for k in range(p):
            nWedge = 0
            Wedgefirst = 0
            Wedgelast = -1
            left = -window
            pleft = 0
            for i in range(1, n):
                left += 1
                if left >= lstart:
                    linidx = k if nWedge == 0 else k + p*Wedge[Wedgefirst]
                    minidx[k, pleft, j] = idx[k, linidx, j]
                    minval[k, pleft, j] = A[k, linidx, j]
                    pleft += 1
                if A[k, i, j] < A[k, i-1, j]:
                    while nWedge:
                        if A[k, i, j] >= A[k, p*Wedge[Wedgelast], j]:
                            if left == Wedge[Wedgefirst]:
                                nWedge -= 1
                                Wedgefirst = (Wedgefirst + 1) % size
                            break
                        nWedge -= 1
                        Wedgelast = (Wedgelast - 1) % size
                else:
                    nWedge += 1
                    Wedgelast = (Wedgelast + 1) % size
                    Wedge[Wedgelast] = i - 1
                    if left == Wedge[Wedgefirst]:
                        nWedge -= 1
                        Wedgefirst = (Wedgefirst + 1) % size
            for i in range(n, imax + 1):
                left += 1
                linidx = k if nWedge == 0 else k + p*Wedge[Wedgefirst]
                minidx[k, pleft, j] = idx[k, linidx, j]
                minval[k, pleft, j] = A[k, linidx, j]
                pleft += 1
                nWedge += 1
                Wedgelast = (Wedgelast + 1) % size
                Wedge[Wedgelast] = n - 1
                if left == Wedge[Wedgefirst]:
                    nWedge -= 1
                    Wedgefirst = (Wedgefirst + 1) % size

    return minval, minidx



def minmaxfilt(A, window=3, outtype='both', shape='valid'):
    if outtype not in ['both', 'minmax', 'maxmin', 'min', 'max']:
        raise ValueError(f"Unknown outtype {outtype}")
    if shape not in ['valid', 'same', 'full']:
        raise ValueError(f"Unknown shape {shape}")

    shapeloc_map = {'valid': 1, 'same': 2, 'full': 3}
    shapeloc = shapeloc_map[shape]

    if np.isscalar(window):
        window = [window] * A.ndim
    else:
        while len(window) < A.ndim:
            window.append(1)

    results = []
    idx_results = []

    # Min Filter
    if outtype != 'max':
        minval = np.copy(A)
        minidx = np.arange(np.prod(A.shape)).reshape(A.shape).astype(np.float64)
        for dim in range(A.ndim):
            shape_dim = minval.shape[dim]
            if window[dim] != 1:
                minval, minidx = lemire_nd_minengine(minval, minidx, window[dim], shapeloc)
            if minval.shape[dim] != shape_dim:
                shape = list(minval.shape)
                shape[dim] = shape_dim
                minval = minval.reshape(shape)
                minidx = minidx.reshape(shape)
        results.append(minval)
        idx_results.append(minidx)

    # Max Filter
    if outtype != 'min':
        maxval = np.copy(A)
        maxidx = np.arange(np.prod(A.shape)).reshape(A.shape).astype(np.float64)
        for dim in range(A.ndim):
            shape_dim = maxval.shape[dim]
            if window[dim] != 1:
                maxval, maxidx = lemire_nd_maxengine(maxval, maxidx, window[dim], shapeloc)
            if maxval.shape[dim] != shape_dim:
                shape = list(maxval.shape)
                shape[dim] = shape_dim
                maxval = maxval.reshape(shape)
                maxidx = maxidx.reshape(shape)
        results.append(maxval)
        idx_results.append(maxidx)

    if len(results) == 1:
        return results[0], idx_results[0]
    return tuple(results + idx_results)

def parameter_sel(img_hazy):
    I = img_hazy
    gray = np.max(I, axis=2)

    # Gaussian filter
    f = gaussian_filter(gray, sigma=10)

    # Laplacian filter
    f3 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    shap = np.abs(convolve2d(gray, f3, mode='same'))
    shap = shap[10:-10, 10:-10]

    # Applying the maximum filter (which is equivalent to the minmaxfilt in MATLAB with 'max' argument)
    shap = maximum_filter(shap, size=20)

    # Apply the Gaussian filter multiple times
    for _ in range(4):
        gray = gaussian_filter(gray, sigma=10)

    gray = gray[10:-10, 10:-10]

    ratio = np.sum(gray) / np.sum(shap)
    alpha = 50
    beta = 0.001
    pro = 1.6

    if ratio > 2.3:
        beta = 0.01
        pro = 2.3

    return alpha, beta, pro
    


def fft_convolution(image, kernel):
    image_padding = len(kernel)//2

    #padding image
    # image_pad = np.pad(image_pad,(((image_padding+1)//2,image_padding//2),((image_padding+1)//2, image_padding//2)), mode='edge')
    image_pad = np.pad(image, image_padding, mode='edge')
    pad_x = image_pad.shape[0] - kernel.shape[0]
    pad_y = image_pad.shape[1] - kernel.shape[1]

    #pad kernel so it is same size as image
    pad_0_low = image_pad.shape[0] // 2 - kernel.shape[0] // 2
    pad_0_high = image_pad.shape[0] - kernel.shape[0] - pad_0_low
    pad_1_low = image_pad.shape[1] // 2 - kernel.shape[1] // 2
    pad_1_high = image_pad.shape[1] - kernel.shape[1] - pad_1_low
    kernel = np.pad(kernel, ((pad_0_low, pad_0_high),(pad_1_low, pad_1_high)), 'constant')
    #move the kernel so center is in the top left corner
    kernel = np.fft.ifftshift(kernel)

    #convert both to fft
    img_fft = np.fft.fft2(image_pad)
    kernel_fft = np.fft.fft2(kernel)

    #multiply 2 fourier matrices
    img = img_fft * kernel_fft
    #inverse fft
    img_inverse = np.fft.ifft2(img)
    #take the real numbers
    output = np.real(img_inverse)
    #slice array to get rid of padding and original size of image
    if image_padding==0:
        return output
    else:
        return output[image_padding:-image_padding, image_padding:-image_padding]

def gradient_weight(I):
    if len(I.shape) == 3 and I.shape[2] == 3:  # If the image has RGB channels
        I = np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale using standard weights

    lambda_val = 10
    f1 = np.array([[1, -1]])  # Reshaped filter
    f2 = np.array([[1], [-1]])  # Reshaped filter

    Gx = fft_convolution(I, f1)
    Gy = fft_convolution(I, f2)

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

def decomposition(I, alpha, ii, beta, gamma):
    alpha = float(alpha)
    ii = float(ii)
    beta = float(beta)
    gamma = float(gamma)
    I = np.clip(I, 0, 1)
    gray = np.mean(I, axis=2)
    H, W, D = I.shape

    weight_x, weight_y = gradient_weight(I)
    f1 = np.array([[1, -1]])
    f2 = np.array([[1], [-1]])
    f4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    gray = rgb2gray(I)
    I_filt = gaussian_filter(gray, 10)
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
        Ix[:, :, channel] = fft_convolution(I[:, :, channel], f1)
        Iy[:, :, channel] = fft_convolution(I[:, :, channel], f2)

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
        for channel in range(I.shape[2]):
            qx[:, :, channel] = fft_convolution(F[:, :, channel], f1) - Ix[:,:,channel]
            qy[:, :, channel] = fft_convolution(F[:, :, channel], f2) - Iy[:,:,channel]

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

        if np.sum(np.abs(prev_F - F)) / (H * W) < 10 ** (-1.2):
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
    
def getDistAirlight(img, air):
    row, col, n_colors = img.shape

    dist_from_airlight = np.zeros((row, col, n_colors), dtype=np.float)
    for color in range(n_colors):
        dist_from_airlight[:, :, color] = img[:, :, color] - air[color]

    return dist_from_airlight


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def air_light(im, dark):
    [h, w] = im.shape[:2]
    image_size = h * w
    numpx = int(max(math.floor(image_size / 1000), 1))
    darkvec = dark.reshape(image_size, 1)
    imvec = im.reshape(image_size, 3)

    indices = darkvec.argsort()
    indices = indices[image_size - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A
    
    
