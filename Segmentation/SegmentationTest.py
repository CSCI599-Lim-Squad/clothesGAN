import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import ImageSegmentation
from ImageSegmentation import SegmentationNN


try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk

tf.reset_default_graph()

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def load_validation(VALIDATION_PATH):
	validation = []
	for file in scandir(VALIDATION_PATH):
		if file.name.endswith('jpg') or file.name.endswith('png') and file.is_file():
			image = scipy.misc.imread(file.path)
			image = scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
			validation.append(image)
	return validation

set = load_validation('997_Train/')


with tf.Session() as sess:
	model = SegmentationNN('combined_model')
	model.batch_size = 1
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess, 'lib/real.ckpt')
	np.set_printoptions(threshold=np.nan)

	n = 246

	generated_image = model.get_one_result(set[n:n+1], sess)
	images = np.concatenate(generated_image)
	images = images[:,:,:,0]
	images = np.reshape(images, (model.batch_size*IMAGE_HEIGHT, IMAGE_WIDTH))
	print(images)
	image = set[n]
	print(image.shape)

	fig = plt.figure(figsize=(10,10))   
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)

	ax1.imshow(image)
	ax2.imshow(images)

	plt.show()
	



