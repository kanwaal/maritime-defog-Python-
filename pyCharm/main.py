# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from decompositionv001 import *

if __name__ == '__main__':
    foggy_image_path = r'..\Test_Data\V_07_01_0019.jpg' # This is relative path which needs to be updated.

    # Read the image using OpenCV
    foggy_image = cv2.imread(foggy_image_path)

    # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
    foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)

    # Get the defogged image
    # defogged_image, F, G = Defogging(foggy_image,air_light = [.12,.17,.26],ii=7)

    # def Defogging(input_img,air_light = [],ii = 7):
    ii = 7
    adjust_fog_removal = 2
    brightness = 0.5
    input_img = foggy_image
    # Convert image from uint8 to float32 (similar to im2double)
    input_img = input_img.astype(np.float32) / 255.0
    alpha = 20000
    beta = 0.1
    gamma = 10

    F, G, _ = decomposition(input_img, alpha, ii, beta, gamma)

    show_images_side_by_side(F, "Glow Free Layer", G, "Glow Layer")

    # Now importing the MATLAB outputs
    mat_contents = scipy.io.loadmat(r'..\mat_data_file\Decomposition.mat') # File path should be updated.
    F_mat = mat_contents['F']
    G_mat = mat_contents['G']
    # print(mse(G_mat,G))
    # print(mae(G_mat, G))
    # print(mae(F_mat, F))
    print("MSE and MAE of Glow layer (G) are {} and {}, respectively".format(mse(G_mat,G), mae(G_mat,G)))
    print("MSE and MAE of Glow Free layer (F) are {} and {}, respectively".format(mse(F_mat.flatten(), F.flatten()), mae(F_mat, F)))

    show_images_side_by_side(F, "Glow Free Layer (Python)", F_mat, "Glow Free Layer (MATLAB)")
    show_images_side_by_side(G, "Glow Layer (Python)", G_mat, "Glow Layer (MATLAB)")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(F)
    # plt.axis('off')
    # plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(G, 'gray')
    # plt.axis('off')
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
