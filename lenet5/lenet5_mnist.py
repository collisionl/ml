import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np


def change(filename):
    image_data_raw = tf.gfile.FastGFile(filename, "rb").read()

    with tf.Session() as sess:
        color_raw = tf.image.decode_jpeg(image_data_raw)
        color_raw = tf.image.resize(color_raw, [28, 28], method=0)
        image_raw = tf.image.rgb_to_grayscale(color_raw)
        image_data = tf.image.convert_image_dtype(image_raw, dtype=tf.float32)
        kk = []
        arrays = np.array(image_raw.eval())
        for i in range(len(arrays)):
            for j in range(len(arrays[i])):
                kk.append(arrays[i][j][0]/255.0)

        raws = [kk]
        raws = np.array(raws)
        return raws

def Weight(shape):
    init = tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float32)
    return tf.Variable(init)

def Bias(shape):
    init = tf.constant(0.1, shape = shape, dtype = tf.float32)
    return tf.Variable(init)

def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = padding)

def pooling(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                          padding = 'SAME')

# read data
mnist = read_data_sets("./MNIST_data", one_hot = True)
sess = tf.InteractiveSession()

# the network
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    x_mat = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    W = Weight([5, 5, 1, 6])
    b = Bias([6])
    conv1 = tf.nn.relu(conv2d(x_mat, W, 'SAME') + b)

with tf.name_scope('pool1'):
    pool1 = pooling(conv1)

with tf.name_scope('conv2'):
    W = Weight([5, 5, 6, 16])
    b = Bias([16])
    conv2 = tf.nn.relu(conv2d(pool1, W, 'VALID') + b)

with tf.name_scope('pool2'):
    pool2 = pooling(conv2)

with tf.name_scope('fc1'):
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    W = Weight([5 * 5 * 16, 120])
    b = Bias([120])
    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W) + b)

with tf.name_scope('fc2'):
    W = Weight([120, 84])
    b = Bias([84])
    fc2 = tf.nn.relu(tf.matmul(fc1, W) + b)

with tf.name_scope('softmax'):
    W = Weight([84, 10])
    b = Bias([10])
    y = tf.nn.softmax(tf.matmul(fc2, W) + b)

resu = tf.argmax(y, 1)
ans = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(ans * tf.log(y))
equal = tf.equal(resu, tf.argmax(ans, 1))
accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

train = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i % 200 == 0:
        print(('At step %d, accuracy is ' % i) ,)
        print(accuracy.eval(feed_dict = {x: batch[0], ans: batch[1]}))
    train.run(feed_dict = {x: batch[0], ans: batch[1]})


print('Accuracy is ',)
print(accuracy.eval(feed_dict = {x: mnist.test.images, ans: mnist.test.labels}))

# x1 = change("./2.jpg")
# print(resu.eval(feed_dict={x: x1}))



