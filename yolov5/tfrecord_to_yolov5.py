import io
import sys

import tensorflow as tf
import numpy as np
import cv2
import time
import os
import argparse
import imgaug as ia
from PIL import Image, ImageDraw

from tensorflow.python.framework.errors_impl import DataLossError


def check_class_number(records):
	try:
		ret = []
		for record in records:
			it = tf.python_io.tf_record_iterator(record)
			_ret = set()
			while True:
				try:
					example = next(it)
				except StopIteration:
					break

				result = tf.train.Example.FromString(example)
				labels = result.features.feature['image/object/class/label'].int64_list.value
				for _label in labels:
					_ret.add(_label)
			ret.append([min(_ret), max(_ret)])

		return ret
	except KeyboardInterrupt:
		print('Stopped.')


def records_to_yolov5(args):
	annotation_class_count = {}
	img_count = 0
	min_h, max_h, min_w, max_w = 9999, 0, 9999, 0
	cnt = 0
	try:
		for tf_file in args.tf_file:
			print(tf_file)
			it = tf.python_io.tf_record_iterator(tf_file)
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
				height = result.features.feature['image/height'].int64_list.value[0]
				width = result.features.feature['image/width'].int64_list.value[0]
				xmin = result.features.feature['image/object/bbox/xmin'].float_list.value
				xmax = result.features.feature['image/object/bbox/xmax'].float_list.value
				ymin = result.features.feature['image/object/bbox/ymin'].float_list.value
				ymax = result.features.feature['image/object/bbox/ymax'].float_list.value
				text = result.features.feature['image/object/class/text'].bytes_list.value			# [b'head', b'person', b'smoke', b'fire', 'helmet']
				label = result.features.feature['image/object/class/label'].int64_list.value		# [0,1,4,3,2]

				# process image
				out = io.BytesIO(raw_image)
				pil_image = Image.open(out)
				if pil_image.mode == 'RGBA':
					pil_image = pil_image.convert('RGB')
				cv_img = np.array(pil_image)

				# save directory
				if not os.path.exists(os.path.join(args.save_dir, 'images')):
					os.makedirs(os.path.join(args.save_dir, 'images'))
				if not os.path.exists(os.path.join(args.save_dir, 'labels')):
					os.makedirs(os.path.join(args.save_dir, 'labels'))

				# save process
				im = Image.fromarray(cv_img)
				if isinstance(im, Image.Image):
					str_cnt = str(cnt).zfill(6)
					image_output_path = '{}/images/{}_{}.jpg'.format(args.save_dir, args.prefix, str_cnt)
					im.save(image_output_path, format='JPEG')

					label_output_path = '{}/labels/{}_{}.txt'.format(args.save_dir, args.prefix, str_cnt)
					with open(label_output_path, 'w') as f:
						for i, lab in enumerate(label):
							_xmin, _xmax = xmin[i], xmax[i]
							_ymin, _ymax = ymin[i], ymax[i]
							x_center = round((_xmin + _xmax) / 2, 6)
							y_center = round((_ymin + _ymax) / 2, 6)
							width_rel = round(_xmax - _xmin, 6)
							height_rel = round(_ymax - _ymin, 6)
							xywh_str = f"{str(lab- args.downclass)} {str(x_center)}  {str(y_center)} {str(width_rel)} {str(height_rel)}\n"
							f.write(xywh_str)
					cnt += 1
				pil_image.close()
				out.close()
				img_count += 1

		print(f'Total {img_count} images.')
		print('==============================================')

		print('==============================================')
	except KeyboardInterrupt:
		print('Stopped.')



def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('tf_file', nargs='+', help='path of tf record file.')
	parser.add_argument('--check_cls', action='store_true', help="check record's label classes min to max.")
	parser.add_argument('-s', '--save_dir', default=None, type=str,
						help='directory to save datasets. you should make train, val directory.'
							 ' And if this path is not exist, it will make dir auto ')
	parser.add_argument('--prefix', default='', type=str, help='Prefix for images & labels file name.')
	parser.add_argument('--downclass', default=0, type=int,
						help='If classes start from not 0, you write minimum value of class number. It will makes '
							 'output labels class number from 0. And you can not know class number distribution, '
							 'execute with --check_cls option.')
	_args = parser.parse_args()
	return _args


def main():
	args = arg_parse()
	tf_records = args.tf_file

	if args.check_cls:
		rets = check_class_number(tf_records)
		for ret, record in zip(rets, tf_records):
			print(f"{record}: {ret}")
		sys.exit(0)

	ret = records_to_yolov5(args)



if __name__ == '__main__':
	main()
