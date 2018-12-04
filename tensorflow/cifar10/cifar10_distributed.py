# encoding:utf-8
import tensorflow as tf
import os 
import glob
import numpy as np
import cv2
import random

label_list = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# tf.logging.set_verbosity(tf.logging.INFO)


char_dict = dict(zip(label_list,range(10)))


flags = tf.app.flags
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '../cifar10', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '172.17.0.9:22221', 'Comma-separated list of hostname:port pairs')
# 三个worker节点
flags.DEFINE_string('worker_hosts', '172.17.0.12:22221,172.17.0.13:22221,172.17.0.14:22221',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS

params = {
    'learning_rate':FLAGS.learning_rate,
    'train_steps': FLAGS.train_steps,
    'batch_size': FLAGS.batch_size,
    'num_epochs':10000,
    'threads': 16}

    
def shuffle_data_and_label(files):
    files.sort()
    labels = [char_dict[file.split('/')[-2]] for file in files]
    data = zip(files,labels)
    data = list(data)
    random.shuffle(data)
    return data

## 生成tfrecords 函数
def encode_to_tfrecord(file_path,tfrecord_path,col=None,row=None):
    files = glob.glob(file_path+'/*/*.jpg')
    data = shuffle_data_and_label(files)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

# shuffled  test data in file stage
    for image_path,label in data:
            img = cv2.imread(image_path)
            height,width,channel = img.shape
            if col and row:
                img = cv2.resize(img,(col,row))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'channel':tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
            }))
            writer.write(example.SerializeToString())
    #     print(label)
    writer.close()

def parser(record):
    features = tf.parse_single_example(record,features={
    'image_raw':tf.FixedLenFeature([],tf.string),
    'label': tf.FixedLenFeature([],tf.int64),
    'height': tf.FixedLenFeature([],tf.int64),
    'width': tf.FixedLenFeature([],tf.int64),
    'channel': tf.FixedLenFeature([],tf.int64),
    })
    
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    channel = tf.cast(features['channel'],tf.int32)
    image = tf.reshape(image,[height,width,channel])
    image = tf.image.resize_images(image,[32,32],method=0)
    return image,label

def inference(x,scope='charaterNet'):
    # 第一个卷积层 + 池化层（32——>16)
    x = tf.reshape(x,[-1,32,32,3])
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层 + 池化层 (16->8)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层 + 池化层 (8->4)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool3, [-1, 4* 4* 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))
    dropout1 = tf.layers.dropout(dense1,0.3,name='dropout1',training=True)
    logits = tf.layers.dense(inputs=dropout1,
                             units=10,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))

    return logits

def train_input_fn(tfrecords_path,params):
#     train_file = tf.train.match_filenames_once(tfrecords_path)
    dataset = tf.data.TFRecordDataset(tfrecords_path,num_parallel_reads=params['threads'])
    dataset = dataset.map(parser,num_parallel_calls=params['threads']).repeat(params['num_epochs'])
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.shuffle(200)     #这里还是会有问题的，如果不是接近整个dataset大小感觉不行
    dataset = dataset.prefetch(400)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
    return dataset


def main(unused_argv):

    ## 获取dataset,如果没有生成tfrecords 生成tfrecords
    train_files = os.path.join(FLAGS.data_dir,'cifar10_image_data/')
    test_files = os.path.join(FLAGS.data_dir,'cifar10_image_data_test/')
    train_tfrecords_path = os.path.join(FLAGS.data_dir,'tfrecords/train_v2.tfrecord')
    test_tfrecords_path = os.path.join(FLAGS.data_dir,'.tfrecords/test_v2.tfrecord')
    if not (os.path.exists(train_tfrecords_path) or os.path.exists(test_tfrecords_path)):
        encode_to_tfrecord(train_files,train_tfrecords_path)
        encode_to_tfrecord(test_files,test_tfrecords_path)


    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print ('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print ('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')


    # 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    

    
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster,
            worker_device="/job:worker/task:%d" % FLAGS.task_index)):
        
        global_step = tf.contrib.framework.get_or_create_global_step()  # 创建纪录全局训练步数变量

        x,labels = train_input_fn(train_tfrecords_path,params)
        logits = inference(x)
        
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels,tf.int32),
                                                    logits=logits)
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_step = optimizer.minimize(loss=loss,
                                        global_step=global_step)

        hooks=[tf.train.StopAtStepHook(last_step=params['train_steps'])]
        
        with tf.train.MonitoredTrainingSession(master=server.target,
                                              is_chief=(FLAGS.task_index == 0),
                                              hooks=hooks) as mon_sess:
            local_step = 0
            while not mon_sess.should_stop():
                _,batch_loss,step = mon_sess.run([train_step,loss, global_step])
                local_step += 1
                if local_step% 100 ==0:
                    print ('Worker %d: traing step %d dome (global step:%d), loss is %f' % (FLAGS.task_index,
                                                                                            local_step, step,batch_loss))

if __name__ == '__main__':
    tf.app.run()