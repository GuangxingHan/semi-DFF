# --------------------------------------------------------
# Semi-DFF
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2018 Tsinghua University
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guangxing Han
# --------------------------------------------------------

import mxnet as mx
import numpy as np


class FeatureL2LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(FeatureL2LossMetric, self).__init__('FeatureL2Loss')

    def update(self, labels, preds):
        feat_dist = preds[4].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[4].size

class L2_RPN_cls_LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(L2_RPN_cls_LossMetric, self).__init__('L2_rpn_cls_loss')

    def update(self, labels, preds):
        feat_dist = preds[0].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[0].size

class L2_RPN_bbox_LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(L2_RPN_bbox_LossMetric, self).__init__('L2_rpn_bbox_loss')

    def update(self, labels, preds):
        feat_dist = preds[1].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[1].size

class L2_cls_LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(L2_cls_LossMetric, self).__init__('L2_cls_loss')

    def update(self, labels, preds):
        feat_dist = preds[2].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[2].size

class L2_bbos_LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(L2_bbos_LossMetric, self).__init__('L2_bbox_loss')

    def update(self, labels, preds):
        feat_dist = preds[3].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[3].size

