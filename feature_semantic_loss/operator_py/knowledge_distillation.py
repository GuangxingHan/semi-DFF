# --------------------------------------------------------
# Semi-DFF
# Copyright (c) 2018 Tsinghua University
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guangxing Han
# --------------------------------------------------------
import mxnet as mx
import numpy as np

class KnowledgeDistillationOperator(mx.operator.CustomOp):
    def __init__(self, temperature, weight):
        super(KnowledgeDistillationOperator, self).__init__()
        self._temperature = temperature
        self._weight = weight

    def forward(self, is_train, req, in_data, out_data, aux):
        student_logits = in_data[0]
        teacher_logits = in_data[1]

        self.student_logits_temp = student_logits / self._temperature
        self.teacher_logits_temp = teacher_logits / self._temperature

        if len(student_logits.shape) == 2:
            self.student_prob = mx.ndarray.SoftmaxActivation(data=self.student_logits_temp, name="student_cls_act")
            self.teacher_prob = mx.ndarray.SoftmaxActivation(data=self.teacher_logits_temp, name="teacher_cls_act")
        elif len(student_logits.shape) == 4:
            self.student_prob = mx.ndarray.SoftmaxActivation(data=self.student_logits_temp, mode="channel", name="student_cls_act")
            self.teacher_prob = mx.ndarray.SoftmaxActivation(data=self.teacher_logits_temp, mode="channel", name="teacher_cls_act")

        student_prob_log = mx.ndarray.log(self.student_prob)
        teacher_prob_log = mx.ndarray.log(self.teacher_prob)
        cross_entropy_loss = - self.teacher_prob * (student_prob_log - teacher_prob_log)

        if len(student_logits.shape) == 2:
            count = self.student_prob.shape[0]
        elif len(student_logits.shape) == 4:
            count = self.student_prob.shape[0] * self.student_prob.shape[2] * self.student_prob.shape[3]

        cross_entropy_loss = mx.ndarray.sum(cross_entropy_loss) / count * self._temperature * self._temperature * self._weight
        self.assign(out_data[0], req[0], cross_entropy_loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        diff = (self.student_prob - self.teacher_prob) * self._temperature
        if len(in_data[0].shape) == 2:
            count = in_data[0].shape[0]
        elif len(in_data[0].shape) == 4:
            count = in_data[0].shape[0] * in_data[0].shape[2] * in_data[0].shape[3]

        diff = diff / count * self._weight
        self.assign(in_grad[0], req[0], diff)


@mx.operator.register('knowledge_distillation')
class KnowledgeDistillationProp(mx.operator.CustomOpProp):
    def __init__(self, temperature, weight):
        super(KnowledgeDistillationProp, self).__init__(need_top_grad=False)
        self._temperature = int(temperature)
        self._weight = float(weight)

    def list_arguments(self):
        return ['student_logits', 'teacher_logits']

    def list_outputs(self):
        return ['cross_entropy_loss']

    def infer_shape(self, in_shape):
        student_prob_shape = in_shape[0]
        teacher_prob_shape = in_shape[1]

        output_shape = (1, )

        return [student_prob_shape, teacher_prob_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return KnowledgeDistillationOperator(self._temperature, self._weight)
