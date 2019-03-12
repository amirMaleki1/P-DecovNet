import tensorflow as tf
import numpy as np
from PIL import Image

def _activation_summary(x):

  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  for l in losses + [total_loss]:

    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def _variable_on_cpu(name, shape, initializer):

  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):

  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def writeImage(image, filename):
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road, Road_marking, Pavement])
    for l in range(0,6):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def storeImageQueue(data, labels, step):
  for i in range(BATCH_SIZE):
    index = 0
    im = data[i]
    la = labels[i]
    im = Image.fromarray(np.uint8(im))
    im.save("batch_im_s%d_%d.png"%(step,i))
    writeImage(np.reshape(la,(120,160)), "batch_la_s%d_%d.png"%(step,i))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_hist(predictions, labels):
  num_class = predictions.shape[3]
  batch_size = predictions.shape[0]
  hist = np.zeros((num_class, num_class))
  for i in range(batch_size):
    hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
  return hist

def print_hist_summery(hist):
  acc_total = np.diag(hist).sum() / hist.sum()
  print ('accuracy = %f'%np.nanmean(acc_total))
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  print ('mean IU  = %f'%np.nanmean(iu))
  for ii in range(hist.shape[0]):
      if float(hist.sum(1)[ii]) == 0:
        acc = 0.0
      else:
        acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
      print(" class # %d accuracy = %f "%(ii, acc))

def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
      hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print(" class # %d accuracy = %f "%(ii,acc))
