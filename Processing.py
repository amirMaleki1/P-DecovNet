import tensorflow as tf

def preprocess(image, label=None, height=360, width=480):

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(height, width, 3))

    if not label == None:
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(height, width, 1))

        return image, label

    return image

def get_filename_list(path):
  fd = open(path)
  image_filenames = []
  label_filenames = []
  filenames = []
  for i in fd:
    i = i.strip().split(" ")
    image_filenames.append(i[0])
    label_filenames.append(i[1])
  return image_filenames, label_filenames


def CamVid_reader(filename_queue):

  image_filename = filename_queue[0]
  label_filename = filename_queue[1]

  imageValue = tf.read_file(image_filename)
  labelValue = tf.read_file(label_filename)

  image_bytes = tf.image.decode_png(imageValue)
  label_bytes = tf.image.decode_png(labelValue)

  image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
  label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

  return image, label

val_dir = "/home/ai/Desktop/SegNet/CamVid/val.txt"
image_dir = "/home/ai/Desktop/SegNet/CamVid/train.txt"
test_dir = "/home/ai/Desktop/SegNet/CamVid/test.txt"
batch_size = 1


image_filenames, label_filenames = get_filename_list(val_dir)

images, labels = CamVid_reader(image_filenames, label_filenames)

images, labels = preprocess(images, label, height=120,width=160)








