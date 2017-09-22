from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_1024_heng
import cv2

### These parameters might be changed ###
input_size = 1024
max_epochs = 150
threshold = 0.49

downscale = cv2.INTER_CUBIC
upscale = cv2.INTER_AREA
### These parameters might be changed ###

orig_width = 1918
orig_height = 1280

batch_size = 12
model_factory = get_unet_128

if input_size == 128:
	batch_size = 24
	model_factory = get_unet_128
elif input_size == 256:
	batch_size = 12
	model_factory = get_unet_256
elif input_size == 512:
	batch_size = 6
	model_factory = get_unet_512
elif input_size == 1024:
	batch_size = 3
	model_factory = get_unet_1024
	downscale = cv2.INTER_LINEAR
	upscale = cv2.INTER_LINEAR

