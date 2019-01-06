import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

epoches = 100
batchsize = 100
index = np.array([13] * batchsize).astype(np.int32)
mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

IDX = tf.placeholder(tf.int32, shape=[batchsize], name='index')
X = tf.placeholder(tf.float32, shape=[batchsize, 784], name='input')
Y = tf.placeholder(tf.float32, shape=[batchsize, 10], name='target')

def conv2d(inputs):
    return tf.layers.conv2d(inputs, filters=3, kernel_size=(2, 28), strides=[1, 1], data_format='channels_last')


def DmaxPooling2d(inputs, idx):
    s = np.array([]).astype(np.float32)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[-1]):
            s = tf.concat([s, [tf.reduce_max(inputs[i, :idx[i], 0, j])],
                           [tf.reduce_max(inputs[i, idx[i]:inputs.shape[1], 0, j])]], axis=0)
    return tf.reshape(s, [batchsize, -1])


def dense(inputs):
    s = tf.layers.dense(inputs=inputs, units=10, activation=tf.nn.relu)
    return s

x_image = tf.reshape(X, shape=[-1, 28, 28, 1])
conv1 = conv2d(x_image)

pool = DmaxPooling2d(conv1, IDX)
# pool = tf.layers.max_pooling2d(conv1, [2,1], 1)
# print(pool.get_shape())
# pool = tf.reshape(pool, [batchsize, 26*3])

# print(pool.get_shape())
logits = dense(pool)

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoches):
        x_input, y_output = mnist.train.next_batch(batchsize)
        _, loss_, acc = sess.run([optimizer, loss, accuracy], feed_dict={X: x_input, Y: y_output, IDX: index})

        print('epoch{0}:\tloss: {1}\taccuracy:{2}'.format(i, loss_, acc))

    # print(type(conv1))
    # output = sess.run(conv1, feed_dict={X: x_input, IDX: index})
    # print(output.shape)
    # print(type(output))
    #
    # output2 = sess.run(pool, feed_dict={X:x_input, IDX:index})
    # print(output2.shape)
    # print(type(output2))
