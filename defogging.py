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
    elif shapeflag == 2:  # SAME_SHAPE
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



def vote_2D(points, points_weight, directions_all, Aall, thres):
    n_directions = directions_all.shape[0]
    accumulator_votes_idx = np.zeros((Aall.shape[0], points.shape[0], n_directions), dtype=bool)
    
    for i_point in range(points.shape[0]):
        for i_direction in range(n_directions):
            idx_to_use = np.where((Aall[:, 0] > points[i_point, 0]) & (Aall[:, 1] > points[i_point, 1]))[0]
            if len(idx_to_use) == 0:
                continue
            
            dist1 = np.sqrt(np.sum(np.power([Aall[idx_to_use, 0]-points[i_point, 0], Aall[idx_to_use, 1]-points[i_point, 1]], 2), axis=0))
            dist1 = dist1/np.sqrt(2) + 1
            
            dist = (-points[i_point, 0]*directions_all[i_direction, 1] +
                    points[i_point, 1]*directions_all[i_direction, 0] +
                    Aall[idx_to_use, 0]*directions_all[i_direction, 1] -
                    Aall[idx_to_use, 1]*directions_all[i_direction, 0])
            
            idx = np.abs(dist) < 2 * thres * dist1
            if not np.any(idx):
                continue

            idx_full = idx_to_use[idx]
            accumulator_votes_idx[idx_full, i_point, i_direction] = True
    
    accumulator_votes_idx2 = np.sum(accumulator_votes_idx.astype(np.uint8), axis=2) >= 2
    accumulator_votes_idx = np.logical_and(accumulator_votes_idx, accumulator_votes_idx2[:, :, np.newaxis])
    
    accumulator_unique = np.zeros(Aall.shape[0])
    for iA in range(Aall.shape[0]):
        idx_to_use = np.where((Aall[iA, 0] > points[:, 0]) & (Aall[iA, 1] > points[:, 1]))[0]
        points_dist = np.sqrt(np.power(Aall[iA, 0] - points[idx_to_use, 0], 2) + np.power(Aall[iA, 1] - points[idx_to_use, 1], 2))
        points_weight_dist = points_weight.flatten()[idx_to_use] * (5 * np.exp(-points_dist) + 1)

#         points_weight_dist = points_weight[idx_to_use] * (5 * np.exp(-points_dist) + 1)
        
        accumulator_unique[iA] = np.sum(points_weight_dist[np.any(accumulator_votes_idx[iA, idx_to_use, :], axis=1)])
    
    Aestimate_idx = np.argmax(accumulator_unique)
    Aout = Aall[Aestimate_idx, :]
    Avote2 = accumulator_unique

    return Aout, Avote2


def generate_Avals(Avals1, Avals2):
    """
    Generate a list of air-light candidates of 2-channels, using two lists of
    values in a single channel each. 
    Aall's shape is (len(Avals1) * len(Avals2), 2).
    """
    Avals1 = np.reshape(Avals1, (-1, 1))
    Avals2 = np.reshape(Avals2, (-1, 1))
    
    A1 = np.kron(Avals1, np.ones((len(Avals2), 1)))
    A2 = np.kron(np.ones((len(Avals1), 1)), Avals2)
    
    Aall = np.hstack((A1, A2))
    
    return Aall
    


def impyramid(A, direction):
    """
    Image pyramid reduction and expansion.
    
    Parameters:
    - A: Input image.
    - direction: 'reduce' or 'expand'
    
    Returns:
    B: Output image.
    """
    
    # Validate direction
    if direction not in ['reduce', 'expand']:
        raise ValueError("Direction must be 'reduce' or 'expand'.")

    M, N = A.shape[:2]

    if direction == 'reduce':
        scaleFactor = 0.5
        outputSize = (int(np.ceil(M / 2)), int(np.ceil(N / 2)))
        kernel = make_piecewise_constant_function(
            [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, float('-inf')],
            [0.0, 0.0625, 0.25, 0.375, 0.25, 0.0625, 0.0]
        )
        kernelWidth = 5
    else:
        scaleFactor = 2
        outputSize = (2 * M - 1, 2 * N - 1)
        kernel = make_piecewise_constant_function(
            [1.25, 0.75, 0.25, -0.25, -0.75, -1.25, float('-inf')],
            [0.0, 0.125, 0.5, 0.75, 0.5, 0.125, 0.0]
        )
        kernelWidth = 3

    # The following code is a simple replacement for imresize in MATLAB
    B = zoom(A, (scaleFactor, scaleFactor, 1), order=1)
    
    # Adjust the size of B to match the expected outputSize
    B = B[:outputSize[0], :outputSize[1]]

    return B

def make_piecewise_constant_function(breakPoints, values):
    """
    Constructs a piecewise constant function and returns a handle to it.
    """
    def piecewise_constant_function(x):
        y = np.zeros_like(x)
        for k in range(x.size):
            yy = 0
            xx = x[k]
            for p in range(len(breakPoints)):
                if xx >= breakPoints[p]:
                    yy = values[p]
                    break
            y[k] = yy
        return y
    
    return piecewise_constant_function


def array_to_image(arr):
    return Image.fromarray(np.uint8(arr))


def rgb2ind(img, m=None, dither='dither'):
    
    if isinstance(img, np.ndarray):
        img = array_to_image(img)
        
    img_arr = np.array(img)

    if m is None:
        indexed_image = img.convert("P", dither=Image.NONE)
        return indexed_image, indexed_image.getpalette()

    elif isinstance(m, int):  # N is given. Use variance minimization quantization
        print(m)
        indexed_image = img.quantize(colors=m, dither=dither != 'nodither')
        return indexed_image, indexed_image.getpalette()

    elif isinstance(m, float):  # TOL is given. Use uniform quantization
        max_colors = 65536
        max_N = int(max_colors ** (1/3)) - 1
        N = round(1 / m)
        N = min(N, max_N)
        step_size = 256 / (N + 1)
        levels = np.arange(0, 256, step_size)
        quantized_arr = np.digitize(img_arr, bins=levels) - 1
        indexed_image = Image.fromarray(np.uint8(quantized_arr))
        return indexed_image, indexed_image.getpalette()

    else:  # MAP is given
        indexed_image = img.quantize(palette=m, dither=dither != 'nodither')
        return indexed_image, indexed_image.getpalette()
        
        
def rgb2ind(img, m=256, dither='dither'):
    
    if isinstance(img, Image.Image):  # If it's a PIL Image, convert to numpy array
        img = np.array(img)
        

    # Case where m is an integer (N is given)
    if isinstance(m, int):
       
        if m <= 0 or m > 65536:
            
            raise ValueError("N must be between 1 and 65536.")

        if m == 1:  # If only one color is asked for, return the average color
           
            map_colors = np.mean(img, axis=(0, 1)).reshape(1, 3)
        else:
       
            # Use variance minimization quantization using KMeans
            pixels = img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=m)
            labels = kmeans.fit_predict(pixels)
            map_colors = kmeans.cluster_centers_
            
            # Dithering is ignored in this version for simplicity
    
        X = labels.reshape(img.shape[:-1])
        return X, map_colors
        
        
def estimate_airlight(gamma, img, Amin=None, Amax=None, N=None, spacing=None, K=None, thres=None):
    print("Starting the estimate_airlight function...")

    # Default value assignments
    if thres is None:
        thres = 0.01
    if spacing is None:
        spacing = 0.02
    if N is None:
        N = 150
    if K is None:
        K = 40
    if Amin is None:
        Amin = [0, 0.05, 0.1]
    if Amax is None:
        Amax = 1

    print("Checking Amin and Amax dimensions...")
    if np.isscalar(Amin):
        Amin = [Amin] * 3
    if np.isscalar(Amax):
        Amax = [Amax] * 3

    print("Reducing image size...")
    # Assuming you have an equivalent Python function for `impyramid`
    img = impyramid(img, 'reduce')
    img = np.power(img, gamma)
    # Assuming you have an equivalent Python function for `rgb2ind`
    img_ind, points = rgb2ind(img, N)
    h, w, _ = img.shape

    print("Removing empty clusters...")
    idx_in_use = np.unique(img_ind)
    idx_to_remove = np.setdiff1d(np.arange(0, points.shape[0]), idx_in_use)
    points = np.delete(points, idx_to_remove, axis=0)
    img_ind_sequential = np.zeros((h, w), dtype=int)
    for kk, idx in enumerate(idx_in_use):
        img_ind_sequential[img_ind == idx] = kk + 1

    print("Counting occurrences of each index...")
    points_weight, _ = np.histogram(img_ind_sequential.ravel(), bins=points.shape[0])
    points_weight = points_weight / (h * w)
    if len(points.shape) > 2:
        points = points.reshape(-1, 3)

    print("Defining arrays of candidate air-light values and angles...")
    angle_list = np.linspace(0, np.pi, K).reshape(-1, 1)
    directions_all = np.column_stack((np.sin(angle_list[:-1]), np.cos(angle_list[:-1])))

    ArangeR = np.arange(Amin[0], Amax[0] + spacing, spacing)
    ArangeG = np.arange(Amin[1], Amax[1] + spacing, spacing)
    ArangeB = np.arange(Amin[2], Amax[2] + spacing, spacing)

    print("Estimating air-light in each pair of color channels...")
    Aall = generate_Avals(ArangeR, ArangeG)
    _, AvoteRG = vote_2D(points[:, 0:2], points_weight, directions_all, Aall, thres)

    Aall = generate_Avals(ArangeG, ArangeB)
    _, AvoteGB = vote_2D(points[:, 1:3], points_weight, directions_all, Aall, thres)

    Aall = generate_Avals(ArangeR, ArangeB)
    _, AvoteRB = vote_2D(points[:, [0, 2]], points_weight, directions_all, Aall, thres)

    print("Finding most probable airlight...")
    max_val = max([np.max(AvoteRB), np.max(AvoteRG), np.max(AvoteGB)])
    AvoteRG2 = AvoteRG / max_val
    AvoteGB2 = AvoteGB / max_val
    AvoteRB2 = AvoteRB / max_val
    

    # Corrected code for A11, A22, and A33:
    A11 = np.tile(AvoteRG2.reshape(len(ArangeG), len(ArangeR)).T[:, :, np.newaxis], (1, 1, len(ArangeB)))
    A22 = np.tile(AvoteRB2.reshape(len(ArangeB), len(ArangeR)).T[:, np.newaxis, :], (1, len(ArangeG), 1))
    A33 = np.tile(AvoteGB2.reshape(len(ArangeB), len(ArangeG)).T[np.newaxis, :, :], (len(ArangeR), 1, 1))
    

    AvoteAll = A11 * A22 * A33
    idx = np.argmax(AvoteAll)
    idx_r, idx_g, idx_b = np.unravel_index(idx, (len(ArangeR), len(ArangeG), len(ArangeB)))
    Aout = [ArangeR[idx_r], ArangeG[idx_g], ArangeB[idx_b]]

    print("Function completed successfully!")
    return Aout



def gradient_weight(I):
    if len(I.shape) == 3 and I.shape[2] == 3:  # If the image has RGB channels
        I = np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale using standard weights
    
    lambda_val = 10
    f1 = np.array([[1, -1]])  # Reshaped filter
    f2 = np.array([[1], [-1]])  # Reshaped filter
    
    Gx = - convolve(I, f1, mode='wrap')
    Gy = - convolve(I, f2, mode='wrap')
    
    ax = np.exp(-lambda_val * np.abs(Gx))
    thx = Gx < 0.01
    ax[thx] = 0
    weight_x = 1 + ax
    
    ay = np.exp(-lambda_val * np.abs(Gy))
    thy = Gy < 0.01
    ay[thy] = 0
    weight_y = 1 + ay
    
    return weight_x, weight_y
    
    
    
def decomposition(I, alpha, ii, beta, gamma):
    I = np.clip(I, 0, 1)
    gray = np.mean(I, 2)
    H, W, D = I.shape

    # Convolutional kernels
    f1 = np.array([[1, -1]])
    f2 = np.array([[1], [-1]])
    f4 = np.array([[0, -1,  0], [-1,  4, -1], [0, -1,  0]])

    # Enhance gradient of I
    weight_x, weight_y = gradient_weight(I)

    I_filt = gaussian_filter(gray, 10)
    delta_I = I - np.repeat(I_filt[:, :, np.newaxis], 3, axis=2)

    # Compute OTFs
    otfFx = psf2otf(f1, [H, W])
    otfFy = psf2otf(f2, [H, W])
    otfL = psf2otf(f4, [H, W])

    fft_double_laplace = np.abs(otfL) ** 2
    fft_double_grad = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        fft_double_grad = np.repeat(fft_double_grad[:, :, np.newaxis], D, axis=2)
        fft_double_laplace = np.repeat(fft_double_laplace[:, :, np.newaxis], D, axis=2)
        weight_x = np.repeat(weight_x[:, :, np.newaxis], D, axis=2)
        weight_y = np.repeat(weight_y[:, :, np.newaxis], D, axis=2)

    F = np.zeros_like(I)
    N = np.zeros_like(I)
    
    # Assuming I is a color image (H x W x 3)
    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)

    for channel in range(I.shape[2]):
        Ix[:, :, channel] = -convolve2d(I[:, :, channel], f1, mode='same', boundary='wrap')
        Iy[:, :, channel] = -convolve2d(I[:, :, channel], f2, mode='same', boundary='wrap')

    Normin_I = np.zeros_like(I)
    for channel in range(I.shape[2]):
        Normin_Ix_channel = np.diff(Ix[:, :, channel], axis=1, prepend=Ix[:, -1:, channel])
        Normin_Iy_channel = np.diff(Iy[:, :, channel], axis=0, prepend=Iy[-1:, :, channel])
        Normin_I[:, :, channel] = Normin_Ix_channel + Normin_Iy_channel


    Denormin_N = gamma + alpha * fft_double_laplace + beta
    Normin_gI = fft_double_laplace * fft2(I)

    i = 0
    while True:
        i += 1
        prev_F = F.copy()
        lambda_val = min(2 ** (ii + i), 10 ** 5)
        Denormin_F = lambda_val * fft_double_grad + alpha * fft_double_laplace + beta

        qx = -convolve_rgb(F, f1) - Ix
        qy = -convolve_rgb(F, f2) - Iy
        qx = np.sign(qx) * np.maximum(np.abs(qx) - weight_x / lambda_val, 0)
        qy = np.sign(qy) * np.maximum(np.abs(qy) - weight_y / lambda_val, 0)

        Normin_q = fft2(np.diff(qx, axis=1, prepend=qx[:, -1:]) + np.diff(qy, axis=0, prepend=qy[-1:, :]))
        Normin_gN = fft_double_laplace * fft2(N)

        FF = (lambda_val * (Normin_I + fft2(Normin_q)) + alpha * (Normin_gI - Normin_gN) + beta * fft2(delta_I - N)) / Denormin_F
        F = np.real(ifft2(FF))

        Normin_F = fft_double_laplace * fft2(F)
        B = fft2(delta_I - F)
        NN = (alpha * (Normin_gI - Normin_F) + beta * B) / Denormin_N
        N = np.real(ifft2(NN))
        print(np.sum(np.abs(prev_F - F)) / (H * W))

        if np.sum(np.abs(prev_F - F)) / (H * W) < 10 ** (1):
            break

    # Normalize F
    for c in range(D):
        Ft = F[:, :, c]
        q = Ft.size
        for k in range(500):
            m = np.sum(Ft[Ft < 0])
            n = np.sum(Ft[Ft > 1] - 1)
            dt = (m + n) / q
            if abs(dt) < 1 / q:
                break
            Ft -= dt
        F[:, :, c] = Ft

    F = np.abs(F)
    F = np.clip(F, 0, 1)
    N = np.clip(N, 0, 1)
    N = np.mean(N, axis=2)
    G = np.abs(I - F)

    return F, G, N


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


def wls_optimization(in_img, data_weight, guidance, lambda_val=0.05):
    """
    Weighted Least Squares optimization solver.
    
    This function is based on the WLS Filter by Farbman, Fattal, Lischinski, and Szeliski.
    """
    small_num = 0.00001
 
    if len(guidance.shape) == 3:
        h, w, _ = guidance.shape
    else:
        h, w = guidance.shape

    
#     h, w = guidance.shape[:2]
    k = h * w
#     guidance = rgb2gray(guidance)
    if len(guidance.shape) == 3:  # Check if it's an RGB image
        guidance = rgb2gray(guidance)

    
    # Compute affinities between adjacent pixels based on gradients of guidance
    dy = np.diff(guidance, axis=0)
    dy = -lambda_val / (np.abs(dy)**2 + small_num)
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
    dy = dy.ravel()
    
    dx = np.diff(guidance, axis=1)
    dx = -lambda_val / (np.abs(dx)**2 + small_num)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
    dx = dx.ravel()
    
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    B = np.stack([dx, dy], axis=-1)
    d = [-h, -1]
    tmp = spdiags(B.T, d, k, k)
    
    ea = dx
    we = np.pad(dx, (h, 0), mode='constant')[:-h]
    so = dy
    no = np.pad(dy, (1, 0), mode='constant')[:-1]
    
    D = -(ea + we + so + no)
    Asmoothness = tmp + tmp.transpose() + spdiags(D, 0, k, k)
    
    # Normalize data weight
    data_weight = data_weight - np.min(data_weight)
    data_weight = data_weight / (np.max(data_weight) + small_num)
    
    # Boundary condition for the top line
    reliability_mask = data_weight[0, :] < 0.6
    in_row1 = np.min(in_img, axis=0).reshape(1, -1)
    if len(guidance.shape) == 3:  # If the guidance image is RGB
        # Modify the mask to be 2D
        reliability_mask_2D = np.any(reliability_mask, axis=2)
        data_weight[0, reliability_mask_2D] = 0.8
        in_img[0, reliability_mask_2D] = in_row1[reliability_mask_2D]
    else:  # If the guidance image is grayscale
        data_weight[0, :] = 0.8
        in_img[0, :] = in_row1


    
    
#     data_weight[0, reliability_mask] = 0.8
#     in_img[0, reliability_mask] = in_row1[0, reliability_mask]

#     in_img[0, reliability_mask] = in_row1[reliability_mask]
    
    Adata = spdiags(data_weight.ravel(), 0, k, k)
    
    A = Adata + Asmoothness
    b = Adata @ in_img.ravel()
    
    out = spsolve(A, b)
    out = out.reshape(h, w)
    
    return out


def non_local_dehazing(img_hazy, air_light):
    h, w, n_colors = img_hazy.shape
    
    # Ensure input has 3 color channels
    if n_colors != 3:
        raise ValueError("Image should have 3 color channels.")
    
    img_hazy_corrected = img_hazy
    
    # Find Haze-lines
    dist_from_airlight = np.zeros((h, w, n_colors))
    for color_idx in range(n_colors):
        dist_from_airlight[:, :, color_idx] = img_hazy_corrected[:, :, color_idx] - air_light[:, :, color_idx]

    # Calculate radius
    radius = np.sqrt(np.sum(dist_from_airlight**2, axis=2))

    # Cluster pixels to haze-lines using KDTree
    dist_unit_radius = dist_from_airlight.reshape((h*w, n_colors))
    dist_norm = np.linalg.norm(dist_unit_radius, axis=1, keepdims=True)
    dist_unit_radius /= dist_norm

    points = load_tesselation_points(1000)  # Placeholder value for n_points
    tree = KDTree(points)
    _, ind = tree.query(dist_unit_radius)

    # Estimating Initial Transmission
    # ...

    # Adjust the shape of the 'ind' array to match the shape of 'radius'
    ind_reshaped = ind.reshape((h, w))

    # Adjust the calculation of K
    # ...
    K = np.array([np.max(radius[ind_reshaped == i]) if np.any(ind_reshaped == i) else 0 for i in range(len(points))])
    # ...


    radius_new = K[ind].reshape(h, w)
    transmission_estimation = np.clip(radius / radius_new, 0.1, 1)
    
    # Regularization
    trans_lower_bound = 1 - np.min(img_hazy_corrected / air_light, axis=2)
    transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)
    lambda_val = 0.1

    # Placeholder for the optimization function
#     transmission = wls_optimization(transmission_estimation, img_hazy)
    transmission = wls_optimization(transmission_estimation, img_hazy, img_hazy, lambda_val)

#     transmission = wls_optimization(transmission_estimation, np.ones_like(transmission_estimation), img_hazy, lambda_val)
    
    # Dehazing
    img_dehazed = np.zeros((h, w, n_colors))
    trans_min = 0.1
    
    for color_idx in range(3):
        img_dehazed[:, :, color_idx] = (img_hazy_corrected[:, :, color_idx] - 
                                transmission * air_light[0, 0, color_idx]) / np.maximum(transmission, trans_min)

#         img_dehazed[:, :, color_idx] = (img_hazy_corrected[:, :, color_idx] - 
#                                         transmission * air_light[color_idx]) / np.maximum(transmission, trans_min)
    
    # Limit pixel values to [0, 1]
    img_dehazed = np.clip(img_dehazed, 0, 1)
    
    return img_dehazed, transmission

def load_tesselation_points(n_points):
    # Read the pre-calculated uniform tesselation of the unit-sphere from a .txt file
    with open(f"TR{n_points}.txt", "r") as file:
        points = np.array([list(map(float, line.strip().split())) for line in file.readlines()])
    return points  
    
def Background(RGB):
    height, width, _ = RGB.shape
    patchSize = 3
    padSize = 1
    JBack = np.zeros((height, width))

    # Pad the image. Here, we pad with large values instead of Inf to find the maximum easily
    imJ = np.pad(RGB, ((padSize, padSize), (padSize, padSize), (0, 0)), mode='constant', constant_values=255)

    for j in range(height):
        for i in range(width):
            patch = imJ[j:j+patchSize, i:i+patchSize, :]
            JBack[j, i] = np.max(patch)

    return JBack

def Compensation(fog_free_layer, G):
    JBack = np.abs(Background(fog_free_layer))
    JBack[JBack > 1] = 1
    JBack[JBack == 0] = 0.001
    
    # Expand dimensions of JBack for broadcasting
    JBack_3D = np.repeat(JBack[:, :, np.newaxis], 3, axis=2)

    GR = G * fog_free_layer
    Gm = GR / JBack_3D
    
    return Gm
    

def Defogging(input_img):
    adjust_fog_removal = 2
    brightness = 0.5

    # Convert to double (equivalent of im2double)
    input_img = input_img.astype(np.float32) / 255.0

    alpha = 20000
    beta = 0.1
    gamma = 10

    _, haze_level, _ = parameter_sel(input_img)
    if haze_level == 0.01:
        ii = 7
    elif haze_level == 0.001:
        ii = 5

    print("size(input_img) : ", input_img.shape)

    F, G, _ = decomposition(input_img, alpha, ii, beta, gamma)

    print("size(F) : ", F.shape)
    print("size(G) : ", G.shape)

#     A = estimate_airlight(adjust_fog_removal, F)
#     A = A.reshape(1, 1, 3)
    A = np.array(estimate_airlight(adjust_fog_removal, F)).reshape(1, 1, 3)

    
    fog_free_layer, _ = non_local_dehazing(F, A)
    
    Gm = Compensation(fog_free_layer, G)

    result = fog_free_layer + brightness * Gm

    gray = np.median(result, axis=-1) * 255  # equivalent of rgb2gray
    if np.median(gray) < 128:
        result = fog_free_layer + Gm

    return result


def imgaussfilt(I, sigma, filter_size=None, padding='replicate', filter_domain='auto'):
    # Define the filter size if not provided
    if filter_size is None:
        filter_size = 2 * np.ceil(2 * sigma) + 1

    # Create Gaussian kernel
    kernel_radius = filter_size // 2
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gx = np.exp(-(x**2) / (2*sigma**2))
    gx /= gx.sum()  # Normalize

    if filter_domain == 'auto':
        # Heuristic to decide between spatial or frequency domain (this is a simple heuristic for illustrative purposes)
        filter_domain = 'spatial' if np.prod(I.shape) * np.prod(gx.shape) < 1e6 else 'frequency'
        
    if filter_domain == 'spatial':
        if padding == 'replicate':
            mode = 'nearest'
        elif padding == 'circular':
            mode = 'wrap'
        elif padding == 'symmetric':
            mode = 'mirror'
        else:
            raise ValueError("Unsupported padding mode.")
        
        I = gaussian_filter(I, sigma=sigma, mode=mode, truncate=kernel_radius/sigma)
        
    elif filter_domain == 'frequency':
        # Apply FFT-based convolution (this is a simplified approach)
        I = fftconvolve(I, gx[:, np.newaxis], mode='same')
        I = fftconvolve(I, gx[np.newaxis, :], mode='same')

    return I


def psf2otf(psf, outSize=None):
    """
    Convert point-spread function to optical transfer function.
    """

    def padlength(psfSize, outSize):
        """
        Determine padding sizes for PSF and OTF.
        """
        if len(psfSize) < len(outSize):
            # Pad psfSize to the same length as outSize
            psfSize = list(psfSize) + [1] * (len(outSize) - len(psfSize))
        return psfSize, outSize

    psf = np.asarray(psf, dtype=float)
    psfSize = psf.shape

    if outSize is None:
        outSize = psfSize
    else:
        outSize = tuple(outSize)

    # Check if outSize is smaller than psfSize
    psfSize, outSize = padlength(psfSize, outSize)
    if any(np.asarray(outSize) < np.asarray(psfSize)):
        raise ValueError("outSize cannot be smaller than psfSize in any dimension.")

    # Pad PSF to outSize
    padSize = np.array(outSize) - np.array(psfSize)
    psf = np.pad(psf, [(0, pad) for pad in padSize], mode='constant')

    # Circularly shift the PSF
    shift = [-size // 2 for size in psfSize]
    psf = np.roll(psf, shift, axis=tuple(range(psf.ndim)))

    # Compute the OTF
    otf = np.fft.fftn(psf)

    # Remove negligible imaginary part
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= len(psfSize) * np.finfo(float).eps:
        otf = np.real(otf)

    return otf


def fft2(x, mrows=None, ncols=None):
    if mrows is not None and ncols is not None:
        x_padded = np.zeros((mrows, ncols), dtype=x.dtype)
        x_rows, x_cols = x.shape
        x_padded[:x_rows, :x_cols] = x
        return np.fft.fft2(x_padded)
    else:
        return np.fft.fft2(x)
    
def convolve_rgb(img, kernel):
    # Separate RGB channels
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Convolve each channel
    R = convolve2d(R, kernel, mode='same', boundary='wrap')
    G = convolve2d(G, kernel, mode='same', boundary='wrap')
    B = convolve2d(B, kernel, mode='same', boundary='wrap')
    
    # Merge the channels back
    return np.stack([R, G, B], axis=-1)


def ifft2(f, m_out=None, n_out=None, symmetry='nonsymmetric'):
    """
    Two-dimensional inverse discrete Fourier transform.
    Equivalent to MATLAB's ifft2 function.
    """
    
    m_in, n_in = f.shape[:2]
    
    if m_out is None:
        m_out = m_in
    if n_out is None:
        n_out = n_in
        
    if m_out != m_in or n_out != n_in:
        out_shape = list(f.shape)
        out_shape[0] = m_out
        out_shape[1] = n_out
        f2 = np.zeros(out_shape, dtype=f.dtype)
        mm = min(m_out, m_in)
        nn = min(n_out, n_in)
        f2[:mm, :nn] = f[:mm, :nn]
        f = f2
    
    if symmetry == 'symmetric':
        f = np.conj(f)
        
    # For 2D FFT, we can simply use the ifft2 from numpy
    x = np.fft.ifft2(f)
    
    return x





