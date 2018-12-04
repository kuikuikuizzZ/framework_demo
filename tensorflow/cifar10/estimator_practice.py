
# coding: utf-8

# In[1]:


import tensorflow as tf
import os 
import glob
import numpy as np
import cv2
label_list = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
tf.logging.set_verbosity(tf.logging.INFO)


# In[2]:


char_dict = dict(zip(label_list,range(10)))


# In[3]:


import random


def shuffle_data_and_label(files):
    files.sort()
    labels = [char_dict[file.split('/')[-2]] for file in files]
    data = zip(files,labels)
    data = list(data)
    random.shuffle(data)
    return data

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


# In[9]:


train_files = './cifar10_image_data/'
test_files = './cifar10_image_data_test/'
train_tfrecords_path = './tfrecords/train_v2.tfrecord'
test_tfrecords_path = './tfrecords/test_v2.tfrecord'
if not (os.path.exists(train_tfrecords_path) and os.path.exists(test_tfrecords_path)):
    encode_to_tfrecord(train_files,train_tfrecords_path)
    encode_to_tfrecord(test_files,test_tfrecords_path)


# In[10]:


# use dataset api
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


# ### 检查batch 



# In[5]:


PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN



def build_estimator(config,params):
    return tf.estimator.Estimator(
        model_fn = model_fn,
        config=config,
        params=params,)

def model_fn(features, labels, mode, params):
    logits = inference(features,mode)
    class_predictions = tf.argmax(logits,axis=-1)
    loss = None
    train_op = None
    eval_metric_ops = {}
    predictions = class_predictions
    
    if mode in (TRAIN,EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels,tf.int32),
            logits=logits)
    if mode == TRAIN:
        train_op=get_train_op_fn(loss,params)
    if mode == EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels,
                    predictions=class_predictions,
                    name='accuracy')
        }
    if mode == 'PREDICT':
        predictions = {
            'classes': class_predictions,
            'probabilities': tf.nn.softmax(logits,name='softmax_tensor')
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
    
    
def inference(x,mode,scope='charaterNet'):
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
    dropout1 = tf.layers.dropout(dense1,0.3,name='dropout1',training=mode==tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout1,
                             units=10,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.03))

    return logits

def get_train_op_fn(loss, params):
    """Get the training Op.

    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (object): Hyper-parameters (needs to have `learning_rate`)

    Returns:
        Training Op
    """
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return train_op


# In[12]:



def train_input_fn(batch_size,tfrecords_path,num_epochs):
#     train_file = tf.train.match_filenames_once(tfrecords_path)
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parser,num_parallel_calls=4).repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(1000)     #这里还是会有问题的，如果不是接近整个dataset大小感觉不行
    dataset = dataset.prefetch(4000)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def test_input_fn(batch_size,tfrecords_path):
#     test_file = tf.train.match_filenames_once(tfrecords_path)
    test_dataset = tf.data.TFRecordDataset(tfrecords_path)
    test_dataset = test_dataset.map(parser).repeat(1)
    test_dataset = test_dataset.batch(batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()
    return test_iterator.get_next()


# In[14]:




params = {
    'learning_rate':0.002,
    'train_steps': 50000,
    'batch_size': 32,
    'num_epochs':100,
}

config = tf.estimator.RunConfig(
    model_dir='./model/',
    save_summary_steps=100,
    log_step_count_steps=100,
    save_checkpoints_steps=500,)

model_estimator = build_estimator(config,params)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda:train_input_fn(params['batch_size'], train_tfrecords_path,params['num_epochs']),
    max_steps=params['train_steps'],)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda:test_input_fn(params['batch_size'], test_tfrecords_path),
    steps=None,
    start_delay_secs=10,
    throttle_secs=30)
params['train_steps']=21000
model_estimator.train(input_fn=lambda:train_input_fn(params['batch_size'], train_tfrecords_path,params['num_epochs']),
                     max_steps=params['train_steps'])


# In[ ]:


predict = model_estimator.predict(input_fn=lambda:test_input_fn(params['batch_size'], test_tfrecords_path))

