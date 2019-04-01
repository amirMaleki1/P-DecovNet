import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from Inputs import *


""" legacy code for tf bug in missing gradient with max_pool_argmax """
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.
EVAL_BATCH_SIZE = 5
BATCH_SIZE = 5
# for CamVid
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def loss(logits, labels):
  """
      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, (-1,NUM_CLASSES))
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):

        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def cal_loss(logits, labels):
    loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
    ]) # class 0~10

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)


def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    inputs = inputT
    
    def ReLu(inputs, train_phase, name):
        return tf.nn.relu(batch_norm_layer(inputs, train_phase, name))
      
    with tf.variable_scope(name) as scope:
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = ReLu(bias, train_phase, scope.name)
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out


def conv_layer(inputT, shape, train_phase, activation=True, Before_ReLu=True, name=None):
    width = shape[0]
    hight = shape[1]
    in_channel = shape[2]
    out_channel = shape[3]
    Input_channel = inputT.get_shape().as_list()[3]

    def ReLu(inputs, train_phase, name):
        x = tf.nn.relu(batch_norm_layer(inputs, train_phase, name))
        return x 

    with tf.variable_scope(name) as scope:
      if Before_ReLu is True:
        biases_Input = _variable_on_cpu('biases_Input', [Input_channel], tf.constant_initializer(0.0))
        biass = tf.nn.bias_add(inputT, biases_Input)
        inputT = ReLu(biass, train_phase, scope.name +'_BeforeRelu')
        
      kernel = _variable_with_weight_decay('ort_weights', [width, hight, in_channel, out_channel], initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = ReLu(bias, train_phase, scope.name)
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out
  

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  strides = [1, stride, stride, 1]
##  layer_name = 'layer%s' % name
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
##    tf.summary.histogram(layer_name + '/deconv', deconv)     
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def max_pool_2x2(x, name=None):
##    layer_name ='layer%s' % name
    with tf.name_scope(name):
         pool, pool_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=None)
    #stride[1,x_movment,y_movment ,1]
##         tf.summary.histogram(layer_name + '/pool', pool)     
    return pool, pool_indices


def inference(images, labels, batch_size, phase_train):
    # norm1
    with tf.name_scope(name="inference"):
         with tf.variable_scope("norm-1"):
              norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
              tf.summary.histogram('/norm-1', norm1)  
    # conv1
         with tf.variable_scope("conv-1"):
              conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
              tf.summary.histogram('/conv-1', conv1)  
##    print(conv1.shape)
    # pool1
##    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
##                           padding='SAME', name='pool1')
         with tf.variable_scope("pool-1"):
              pool1, pool1_indices = max_pool_2x2(conv1, name="pool1")
              tf.summary.histogram('/pool-1', pool1) 
##    print(pool1.shape)
    # conv2
         with tf.variable_scope("conv-2"):
              conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
              tf.summary.histogram('/conv-2', conv2) 

    # pool2
##    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
##                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
         with tf.variable_scope("pool-2"):
              pool2, pool2_indices = max_pool_2x2(conv2, name="pool2")
              tf.summary.histogram('/pool-2', pool2) 
##    print(pool2.shape)
    # conv3
         with tf.variable_scope("conv-3"):
              conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")
              tf.summary.histogram('/conv-3', conv3) 

    # pool3
##    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
##                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')
         with tf.variable_scope("pool-3"):
              pool3, pool3_indices = max_pool_2x2(conv3, name="pool3")
              tf.summary.histogram('/pool-3', pool3) 
##    print(pool3.shape)
    # conv4
         with tf.variable_scope("conv-4"):
              conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")
              tf.summary.histogram('/conv-4', conv4) 

    # pool4
##    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
##                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')
         with tf.variable_scope("pool-4"):
              pool4, pool4_indices = max_pool_2x2(conv4, name="pool4")
              tf.summary.histogram('/pool-4', pool4) 
              print('pool4 = ',pool4.get_shape())
              
         """ End of encoder """


         """ Pyramid Pooling Module"""
          
          
         with tf.variable_scope('PyP', dtype=tf.float32):
              rcu01 = conv_layer(images, [3, 3, images.get_shape().as_list()[3], 64], phase_train, activation=True, Before_ReLu=True, name='RCU01')
              rcu11 = tf.add(rcu01, conv_layer(rcu01, [3, 3, rcu01.get_shape().as_list()[3], 64], phase_train, name='RCU11'))
              print('pool11 =' , images.get_shape() , ' and rcu11 =', rcu11.get_shape())
                
              rcu02 = conv_layer(pool1, [3, 3, pool1.get_shape().as_list()[3], 32], phase_train, activation=True, Before_ReLu=True, name='RCU02')
              rcu12 = tf.add(pool1, conv_layer(rcu02, [3, 3, rcu02.get_shape().as_list()[3], 64], phase_train, name='RCU12'))

              rcu03 = conv_layer(pool2, [3, 3, pool2.get_shape().as_list()[3], 32], phase_train, activation=True, Before_ReLu=True, name='RCU03')
              rcu13 = tf.add(pool2, conv_layer(rcu03, [3, 3, rcu03.get_shape().as_list()[3], 64], phase_train, name='RCU13'))
              
              rcu04 = conv_layer(pool3, [3, 3, pool3.get_shape().as_list()[3], 32], phase_train, activation=True, Before_ReLu=True, name='RCU04')
              rcu14 = tf.add(pool3, conv_layer(rcu04, [3, 3, rcu04.get_shape().as_list()[3], 64], phase_train, name='RCU14'))
              
              rcu05 = conv_layer(pool4, [3, 3, pool4.get_shape().as_list()[3], 32], phase_train, activation=True, Before_ReLu=True, name='RCU05')
              rcu15 = tf.add(pool4, conv_layer(rcu05, [3, 3, rcu05.get_shape().as_list()[3], 64], phase_train, name='RCU15'))

              
              pypool1 = tf.image.resize_bilinear(rcu11, [120, 160])
              pypool2 = tf.image.resize_bilinear(rcu12, [120, 160])
              pypool3 = tf.image.resize_bilinear(rcu13, [120, 160])
              pypool4 = tf.image.resize_bilinear(rcu14, [120, 160])
              pypool5 = tf.image.resize_bilinear(rcu15, [120, 160])
              print("PYpool4" , pypool4.get_shape())
              pyp1 = tf.multiply(pypool1, pypool2, name='pyp1')
              pyp2 = tf.multiply(pyp1, pypool3, name='pyp2')
              pyp3 = tf.multiply(pyp2, pypool4, name='pyp3')
              pyp4 = tf.multiply(pyp3, pypool5, name='pyp4')
##              pyp = tf.stack(tf.reduce_mean([pypool1, pypool1, pypool1, pypool1, pypool1], 0))
              print("pyp.shape = " , pyp1.get_shape())
              print("pyp4.shape = " , pyp4.get_shape())
              pyp_conv = conv_layer_with_bn(pyp4, [7, 7, pyp4.get_shape().as_list()[3], 64], phase_train, name='pyp_conv')
              crp = tf.nn.relu(pyp_conv , name='crp')

         pool_add_rcu = tf.add(tf.image.resize_bilinear(crp,[8, 10]), pool4)
         print("pool_add_rcuF = " , pool_add_rcu.get_shape())



         
         """ start upsample """

    # upsample4
    # Need to change when using different dataset out_w, out_h

    
         with tf.variable_scope("upsample-4"):
              upsample4 = deconv_layer(pool_add_rcu, [2, 2, 64, 64], [batch_size, 15, 20, 64], 2, "up4")
              tf.summary.histogram('/upsample4', upsample4) 
    # decode 4
         with tf.variable_scope("decode-4"):
              conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")
              tf.summary.histogram('/conv_decode4', conv_decode4) 

    # upsample 3

         with tf.variable_scope("upsample-3"):
              upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 30, 40, 64], 2, "up3")
              tf.summary.histogram('/upsample3', upsample3) 
    # decode 3
         with tf.variable_scope("decode-3"):
              conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")
              tf.summary.histogram('/conv_decode3', conv_decode3) 

    # upsample2

         with tf.variable_scope("updample-2"):
              upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 60, 80, 64], 2, "up2")
              tf.summary.histogram('/upsample2', upsample2) 
    # decode 2
         with tf.variable_scope("decode-2"):
              conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")
              tf.summary.histogram('/conv_decode2', conv_decode2) 

    # upsample1

         with tf.variable_scope("upsample-1"):
              upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 120, 160, 64], 2, "up1")
              tf.summary.histogram('/upsample1', upsample1) 
    # decode1
         with tf.variable_scope("decode-1"):
              print('upsample = ', upsample1.get_shape())
              conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
              tf.summary.histogram('/conv_decode1', conv_decode1) 
         """ end of Decode """

         """ Start Classify """
    # output predicted class number (6)
         with tf.variable_scope('conv_classifier') as scope:
              kernel = _variable_with_weight_decay('weights',
                                                    shape=[1, 1, 64, NUM_CLASSES],
                                                    initializer=msra_initializer(1, 64),
                                                    wd=0.0005)
              conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
              biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
              conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
         logit = conv_classifier
         loss = cal_loss(conv_classifier, labels)
         tf.summary.histogram(scope.name, loss)     
    return loss, logit


"""Function of Train Step"""

def train(total_loss, global_step):
    total_sample = 274
    num_batches_per_epoch = 274/1
    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

"""Function of Test Step"""

def test(FLAGS):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir # /home/ai/Desktop/pdecovnet/Logs
  test_dir = FLAGS.test_dir # /home/ai/Desktop/pdecovnet/CamVid/train.txt
  test_ckpt = FLAGS.testing
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # testing should set BATCH_SIZE = 1
  Batch_size = 1

  image_filenames, label_filenames = get_filename_list(test_dir)

  test_data_node = tf.placeholder(
        tf.float32,
        shape=[Batch_size, image_h, image_w, image_c])

  test_labels_node = tf.placeholder(tf.int64, shape=[Batch_size, 120, 160, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = inference(test_data_node, test_labels_node, Batch_size, phase_train)

  pred = tf.argmax(logits, dimension=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, test_ckpt )

    images, labels = get_all_test_data(image_filenames, label_filenames)

    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):

      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      # output_image to verify
      if (FLAGS.save_image):
          writeImage(im[0], 'testing_image.png')

      hist += get_hist(dense_prediction, label_batch)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))

def training(FLAGS, is_finetune=False):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir   #"/home/ai/Desktop/pdecovnet/logs"
  image_dir = FLAGS.image_dir #"/home/ai/Desktop/pdecovnet/CamVid/train.txt"
  val_dir = FLAGS.val_dir     #"/home/ai/Desktop/pdecovnet/CamVid/val.txt"
  finetune_ckpt = FLAGS.finetune
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # should be changed if your model stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  image_filenames, label_filenames = get_filename_list(image_dir)
  val_image_filenames, val_label_filenames = get_filename_list(val_dir)

  with tf.Graph().as_default():

    train_data_node = tf.placeholder( tf.float32, shape=[batch_size, image_h, image_w, image_c])

    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
    images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)

    val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size)

    # Build a Graph that computes the logits predictions from the inference model.
    loss, eval_prediction = inference(train_data_node, train_labels_node, batch_size, phase_train)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = train(loss, global_step)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      if (is_finetune == True):
          saver.restore(sess, finetune_ckpt )
      else:
          init = tf.global_variables_initializer()
          sess.run(init)

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # Summery placeholders
      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

      for step in range(startstep, startstep + max_steps):
        image_batch ,label_batch = sess.run([images, labels])
        # since we still use mini-batches in validation, still set bn-layer phase_train = True
##        print(image_batch)
        feed_dict = {
          train_data_node: image_batch,
          train_labels_node: label_batch,
          phase_train: True
        }
        start_time = time.time()

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(eval_prediction, feed_dict=feed_dict)
          per_class_acc(pred, label_batch)

        if step % 100 == 0:
          print("start validating.....")
          total_val_loss = 0.0
          hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
          for test_step in range(0,20):
##            print(test_step)
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += get_hist(_val_pred, val_labels_batch)
          print("val loss: ", total_val_loss / 20)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          print_hist_summery(hist)
          print(" end validating.... ")

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)
