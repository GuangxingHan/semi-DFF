# --------------------------------------------------------
# Semi-DFF
# Copyright (c) 2018 Tsinghua University
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guangxing Han
# --------------------------------------------------------

import numpy as np
import mxnet as mx
import _init_paths
import cv2
import time
import argparse
import logging
import pprint
import os
import sys
import shutil
from config.config import config, update_config


def parse_args():
	parser = argparse.ArgumentParser(description='Train semi-DFF network')
	# general
	parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

	args, rest = parser.parse_known_args()
	# update config
	update_config(args.cfg)

	# training
	parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
	args = parser.parse_args()
	return args


args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


from utils.image import get_absolute_pair_image
from mxnet.executor_manager import _split_input_slice


class ImagenetVIDIter(mx.io.DataIter):
	def __init__(self, imagedb, batch_size, cfg, shuffle=config.TRAIN.SHUFFLE, ctx=[mx.gpu(0)]):
		self.imagedb = imagedb
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.ctx = ctx
		self.cfg = cfg
		self.data_name = ['data', 'data_ref']
		self.label_name = None # ['NONE']

		# infer properties from imagedb
		self.size = len(imagedb)
		self.index = np.arange(self.size)
		self.cur = 0
		self.data = None
		self.label = None

		# get first batch to fill in provide_data and provide_label
		self.reset()
		self.get_batch_individual()

	@property
	def provide_data(self):
		return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

	@property
	def provide_label(self):
		return [None for i in xrange(len(self.data))] # [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.label))] # None # [None for i in xrange(len(self.data))]

	@property
	def provide_data_single(self):
		return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

	@property
	def provide_label_single(self):
		return None # [(k, v.shape) for k, v in zip(self.label_name, self.label[0])] # None # []

	def reset(self):
		self.cur = 0
		if self.shuffle:
			np.random.shuffle(self.index)

	def iter_next(self):
		return self.cur + self.batch_size <= self.size

	def next(self):
		if self.iter_next():
			self.get_batch_individual()
			self.cur += self.batch_size
			return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(), provide_data=self.provide_data, provide_label=self.provide_label)
		else:
			raise StopIteration

	def getindex(self):
		return self.cur / self.batch_size

	def getpad(self):
		if self.cur + self.batch_size > self.size:
			return self.cur + self.batch_size - self.size
		else:
			return 0

	def get_batch_individual(self):
		cur_from = self.cur
		cur_to = min(cur_from + self.batch_size, self.size)
		imagedb = [self.imagedb[self.index[i]] for i in range(cur_from, cur_to)]
		# decide multi device slice
		work_load_list = [1] * len(self.ctx)
		slices = _split_input_slice(self.batch_size, work_load_list)
		rst = []
		for idx, islice in enumerate(slices):
			sub_imagedb = [imagedb[i] for i in range(islice.start, islice.stop)]
			rst.append(self.parfetch(sub_imagedb))
		all_data = [_['data'] for _ in rst]
		# all_label = [_['label'] for _ in rst]
		self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
		self.label = [None for data in all_data]

	def parfetch(self, sub_imagedb):
		assert len(sub_imagedb) == 1, 'Single batch only'
		imgs, ref_imgs, eq_flags, _ = get_absolute_pair_image(sub_imagedb, self.cfg)
		assert eq_flags[0] == 0, 'absolute different images only'
		im_array = imgs[0]
		ref_im_array = ref_imgs[0]

		data = {'data': im_array, 'data_ref': ref_im_array}
		return {'data': data} # {'data': data, 'label': {'NONE':np.empty((0, 5), dtype=np.float32)}}


from symbols import *
from core import callback, metric
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler


def train_feature_distance_net(args, ctx, pretrained, pretrained_flow, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
	# ==============prepare logger==============
	logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
	prefix = os.path.join(final_output_path, prefix)

	# ==============load symbol==============
	shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
	sym_instance = eval(config.symbol + '.' + config.symbol)()
	if config.TRAIN.G_type == 0:
		sym = sym_instance.get_train_feature_distance_symbol(config)
	elif config.TRAIN.G_type == 1:
		sym = sym_instance.get_train_feature_distance_symbol_res(config)

	# ==============setup multi-gpu==============
	batch_size = len(ctx)
	input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

	# ==============print config==============
	pprint.pprint(config)
	logger.info('training config:{}\n'.format(pprint.pformat(config)))

	# ==============load dataset and prepare imdb for training==============
	image_sets = [iset for iset in config.dataset.image_set.split('+')]
	roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path, flip=config.TRAIN.FLIP) for image_set in image_sets]
	roidb = merge_roidb(roidbs)
	roidb = filter_roidb(roidb, config)
	train_iter = ImagenetVIDIter(roidb, input_batch_size, config, config.TRAIN.SHUFFLE, ctx)

	# infer max shape
	max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
					('data_ref', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
	print 'providing maximum shape', max_data_shape

	data_shape_dict = dict(train_iter.provide_data_single)
	pprint.pprint(data_shape_dict)
	sym_instance.infer_shape(data_shape_dict)

	# ==============init params==============
	if config.TRAIN.RESUME:
		print('continue training from ', begin_epoch)
		arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
	else:
		arg_params, aux_params = load_param(pretrained, epoch, convert=True)
		# arg_params_flow, aux_params_flow = load_param(pretrained_flow, epoch, convert=True)
		# arg_params.update(arg_params_flow)
		# aux_params.update(aux_params_flow)
		sym_instance.init_weight(config, arg_params, aux_params)

	# ==============check parameter shapes==============
	# sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

	# ==============create solver==============
	fixed_param_prefix = config.network.FIXED_PARAMS
	data_names = train_iter.data_name
	label_names = train_iter.label_name

	mod = MutableModule(sym, data_names=data_names, label_names=label_names,
						logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
						max_label_shapes=None, fixed_param_prefix=fixed_param_prefix)

	if config.TRAIN.RESUME:
		mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

	# ==============optimizer==============
	optimizer_params={
		'learning_rate': 0.00005,
	}

	if not isinstance(train_iter, PrefetchingIter):
		train_iter = PrefetchingIter(train_iter)

	batch_end_callback = callback.Speedometer(train_iter.batch_size, frequent=args.frequent)
	epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix)]

	feature_L2_loss = metric.FeatureL2LossMetric(config)
	eval_metrics = mx.metric.CompositeEvalMetric()
	eval_metrics.add(feature_L2_loss)

	mod.fit(train_iter, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
			batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
			optimizer='RMSprop', optimizer_params=optimizer_params,
			arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch, 
			initializer=mx.init.Normal(0.02), allow_missing=True)


def main():
	print('Called with argument:', args)
	ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
	train_feature_distance_net(args, ctx, config.network.pretrained, config.network.pretrained_flow, config.network.pretrained_epoch, config.TRAIN.model_prefix,
								config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)


if __name__ == '__main__':
	main()
