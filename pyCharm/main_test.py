from df_v0600 import *

if __name__ == '__main__':
    foggy_image_path = r'.\Test_Data\V_07_01_0019.jpg' # This is relative path which needs to be updated.

    # Read the image using OpenCV
    foggy_image = cv2.imread(foggy_image_path)

    # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
    foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB)

    # Get the defogged image
    defogged_image = Defogging(foggy_image,air_light = [.12,.17,.26],ii=7)
	mat_contents = scipy.io.loadmat(r'.\mat_data_file\df.mat')
	df = mat_contents['df']
	print("MSE and MAE of Gm are {} and {}, respectively".
		  format(mse(df.flatten(),defogged_image.flatten()), mae(df,defogged_image)))
    show_images_side_by_side_3(foggy_image, 'input', defogged_image, 'Python', df, 'MATLAB')
