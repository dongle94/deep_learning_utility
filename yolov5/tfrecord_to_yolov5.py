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


def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('tf_file', nargs='+', help='path of tf record file.')
	parser.add_argument('-d', '--display_image', action='store_true', help='display images.')
	parser.add_argument('-v', '--verbose', action='store_true', help='print annotations per image.')
	parser.add_argument('-s', '--save_image_dir', default=None, type=str, help='directory to save image.')
	parser.add_argument('-p', '--polygon', action='store_true', help='show polygon as masked image.')
	parser.add_argument('--check_cls', action='store_true', help="check record's label classes min to max.")
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



if __name__ == '__main__':
	main()
