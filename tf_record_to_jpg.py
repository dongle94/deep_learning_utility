import io
import numpy as np
import cv2
import time
import os

import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.python.framework.errors_impl import DataLossError

from common import image_util, coord





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# Function from common/ui_utils.py
def show_image(img):
	import matplotlib.pyplot as plt
	from six import string_types
	import scipy.misc

	npimg = img
	if isinstance(img, Image.Image):
		npimg = np.array(img)
	elif isinstance(img, string_types):
		npimg = scipy.misc.imread(img)

	if npimg.dtype == np.uint8:
		npimg = npimg.astype(np.float32)/255.

	if npimg.ndim == 2:
		plt.imshow(npimg, cmap='gray')
	else:
		plt.imshow(npimg)
	mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()
	plt.show()


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('tf_file', nargs='+', help='path of tf record file.')
	parser.add_argument('-d', '--display_image', action='store_true', help='display images.')
	parser.add_argument('-v', '--verbose', action='store_true', help='print annotations per image.')
	parser.add_argument('-s', '--save_image_dir', default=None, type=str, help='directory to save image.')
	parser.add_argument('-p', '--polygon', action='store_true', help='show polygon as masked image.')
	args = parser.parse_args()

	try:
		annotation_class_count = {}
		img_count = 0
		min_h, max_h, min_w, max_w = 9999, 0, 9999, 0
		#print(args.tf_file)
		for tf_file in args.tf_file:
			it = tf.python_io.tf_record_iterator(tf_file)
			#print(len(list(it)))
			while True:
				try:
					example = next(it)
				except DataLossError:
					print('ERROR - Data loss. {} index: {}'.format(tf_file, img_count))
					break
				except StopIteration:
					break
				result = tf.train.Example.FromString(example)
				raw_image = result.features.feature['image/encoded'].bytes_list.value[0]
				file_name = result.features.feature['image/filename'].bytes_list.value[0]
				img_format = result.features.feature['image/format'].bytes_list.value[0]
				height = result.features.feature['image/height'].int64_list.value[0]
				width = result.features.feature['image/width'].int64_list.value[0]
				xmin = result.features.feature['image/object/bbox/xmin'].float_list.value
				xmax = result.features.feature['image/object/bbox/xmax'].float_list.value
				ymin = result.features.feature['image/object/bbox/ymin'].float_list.value
				ymax = result.features.feature['image/object/bbox/ymax'].float_list.value
				text = result.features.feature['image/object/class/text'].bytes_list.value
				label = result.features.feature['image/object/class/label'].int64_list.value
				polygon = result.features.feature['image/object/mask'].bytes_list.value
				source_id = result.features.feature['image/source_id'].bytes_list.value[0]
				print(file_name, img_format, height, width, xmin, xmax, ymin, ymax, label, source_id)

				# image create from tfrecord byte
				out = io.BytesIO(raw_image)

				pil_image = Image.open(out)
				real_format = pil_image.format.lower()

				bboxes = []
				boxes_in_image = {}
				cv_img = None
				length_box = 0

				mask_image = None
				if args.display_image or args.save_image_dir:
					cv_img = np.array(pil_image)

					if args.polygon:
						polygon_mask_added = None
						for p in polygon:
							o = io.BytesIO(p)
							polygon_image = Image.open(o)
							polygon_mask = np.array(polygon_image)
							print(np.max(polygon_mask), np.unique(polygon_mask))
							print(polygon_mask.shape, polygon_mask)
							exit()
							o.close()

							if polygon_mask_added is None:
								polygon_mask_added = polygon_mask
							else:
								polygon_mask_added += polygon_mask
						if polygon_mask_added is not None:
							if pil_image.mode == 'RGBA':
								polygon_mask_added = cv2.cvtColor(polygon_mask_added, cv2.COLOR_GRAY2RGBA)
								polygon_mask_added[..., 3] = 1
							elif pil_image.mode == 'RGB':
								pass
								polygon_mask_added = cv2.cvtColor(polygon_mask_added, cv2.COLOR_GRAY2RGB)
							else:
								print(f"unknown image mode: {pil_image.mode}")

							mask_image = cv_img * polygon_mask_added
							mask_image[..., 0] = np.where(polygon_mask_added[..., 0] == 0, cv_img[..., 0], cv_img[..., 0]*0.4)
							mask_image[..., 1] = np.where(polygon_mask_added[..., 1] == 0, cv_img[..., 1], cv_img[..., 1]*1)
							mask_image[..., 2] = np.where(polygon_mask_added[..., 2] == 0, cv_img[..., 2], cv_img[..., 2]*0.3)

							pil_image = Image.fromarray(mask_image)
				
				for x, y, x2, y2, txt, lab in zip(xmin, ymin, xmax, ymax, text, label):
					class_text = f'{lab}:{txt.decode()}'
					if args.display_image:
						drawing = ImageDraw.Draw(pil_image)
						drawing.rectangle(((width*x, height*y), (width*x2, height*y2)), outline='red')
						# font = ImageFont.truetype("arial",size=12)
						drawing.text((width*x, height*y), f' {class_text}', fill='red')

					if class_text not in boxes_in_image:
						boxes_in_image[class_text] = 1
					else:
						boxes_in_image[class_text] += 1

					if class_text not in annotation_class_count:
						annotation_class_count[class_text] = 1
					else:
						annotation_class_count[class_text] += 1

					if args.save_image_dir:
						point = ((x, y), (x2, y2))
						box = coord.BBox.from_relative_points(point, cv_img.shape)
						box.class_name = class_text
						bboxes.append(box)
					length_box += 1

				tmp_str = ''
				for k, v in boxes_in_image.items():
					tmp_str = f'{tmp_str}({k}  count:{v}),'
				if args.verbose:
					image_size = f'{width}x{height}'
					print_img = f'{img_count+1:<5}:{image_size:<9} {str(img_format.decode())}/{real_format:5}| boxes:{length_box:<3}| {tmp_str} '
					print(print_img)

				min_h = min(min_h, height)
				max_h = max(max_h, height)
				min_w = min(min_w, width)
				max_w = max(max_w, width)

				if args.save_image_dir:
					def func(bbox):
						return bbox.class_name
					if mask_image is not None:
						cv_img = mask_image
					image_util.save_image_file_with_boxes(cv_img, bboxes, args.save_image_dir, func)

				if args.display_image:
					npimage = image_util.get_numpy_image(pil_image)
					show_image(npimage)


				pil_image.close()
				out.close()
				img_count += 1
		print(f'Total {img_count} images.')
		print('===============class information==============')
		for res in sorted(annotation_class_count.items(), key=lambda x: int(x[0].split(':', 1)[0])):
			print(f'class {res[0]:<15} {res[1]}')
		print('==============================================')
		print(f'min width: {min_w}, min height: {min_h}, max_width: {max_w}, max_height: {max_h}')
	except KeyboardInterrupt:
		print('Stopped.')


if __name__ == '__main__':
	main()
