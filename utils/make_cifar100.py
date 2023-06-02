import numpy as np
import tensorflow as tf
# raw_dataset = tf.data.TFRecordDataset("out.tfrecords")

# for raw_record in raw_dataset.take(2):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
#     quit()
import os

def write(d_name, s_idx, e_inx, idx):
  samples = np.load('tmp/CIFAR-100-C/' + d_name,allow_pickle=True)[s_idx:e_inx]
  labels = np.load('tmp/CIFAR-100-C/labels.npy')[s_idx:e_inx]
  d_name = d_name.replace(".npy","")
  print(len(samples))
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  directory = "tmp/tensorflow_datasets/cifar100_corrupted/"

  if not os.path.exists(directory):
    os.makedirs(directory)

  with tf.io.TFRecordWriter(directory + d_name + "-" + str(idx) + ".tfrecords") as record_writer:
      for i in range(len(samples)):
          example = tf.train.Example(features=tf.train.Features(
            feature={
              'image': _bytes_feature(samples[i].tobytes()),
              'label': _int64_feature(labels[i])
            }))
          record_writer.write(example.SerializeToString())

directory = "tmp/CIFAR-100-C/"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if "README.txt" in f.split("/")[2]:
      continue
    if "labels.npy" in f.split("/")[2]:
      continue
    if os.path.isfile(f):
        write(f.split("/")[2], None, 10000, 1)
        write(f.split("/")[2], 10000, 20000, 2)
        write(f.split("/")[2], 20000, 30000, 3)
        write(f.split("/")[2], 30000, 40000, 4)
        write(f.split("/")[2], 40000, None, 5)