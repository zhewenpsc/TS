# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


# TODO(robieta): See if some version of this can be rolled into TPUEstimator.
def construct_scalar_host_call(metric_dict, model_dir):
  # type: (dict, str) -> (function, list)
  metric_names = list(metric_dict.keys())

  def host_call_fn(gs, *args):
    """Training host call. Creates scalar summaries for training metrics.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the
    model to the `metric_fn`, provide as part of the `host_call`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `host_call`.

    Args:
      gs: `Tensor with shape `[batch]` for the global_step
      *args: Remaining tensors to log.

    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = gs[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          tf.contrib.summary.scalar(name, args[i][0], step=gs)

        return tf.contrib.summary.all_summary_ops()

  # To log the current learning rate, and gradient norm for Tensorboard, the
  # summary op needs to be run on the host CPU via host_call. host_call
  # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
  # dimension. These Tensors are implicitly concatenated to
  # [params['batch_size']].
  gs_t = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

  return host_call_fn, [gs_t] + other_tensors

def sleep_to_clear_queues():
  """Sleep for a minute if TPUs are used.

  There is currently an issue with TPUs where starting a train or evaluation
  before all of the TPU queues have cleared causes the TPU to freeze. This
  is a temporary workaround until the issue can be properly resolved.
  """
  tf.logging.info("Sleeping to allow TPU queues to clear.")
  time.sleep(60)
