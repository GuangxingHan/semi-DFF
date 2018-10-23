# --------------------------------------------------------
# Semi-DFF
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
        feat_dist = preds[0].asnumpy()[0]

        self.sum_metric += feat_dist # sum
        self.num_inst += preds[0].size
