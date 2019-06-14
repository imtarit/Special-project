import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]

def vgg16():
    """
    load variable from npy to build the VGG

    :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
    """
    inputData = tf.placeholder(tf.float32, [None, 150528], name='rgb')
    classLabels = tf.placeholder(tf.float32, [None, 8], name='lbl')

    rgb = tf.reshape(inputData, [-1,224,224,3])
    lbl = classLabels

    # print("build model started")
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
    
    conv1_1 = conv_layer(bgr, 3, 64, name = "conv1_1")
    conv1_2 = conv_layer(conv1_1, 64, 64, name = "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, 64, 128, name = "conv2_1")
    conv2_2 = conv_layer(conv2_1, 128, 128, name ="conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, 128, 256, name ="conv3_1")
    conv3_2 = conv_layer(conv3_1, 256, 256, name ="conv3_2")
    conv3_3 = conv_layer(conv3_2, 256, 256, name ="conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv_layer(pool3, 256, 512, name ="conv4_1")
    conv4_2 = conv_layer(conv4_1, 512, 512, name ="conv4_2")
    conv4_3 = conv_layer(conv4_2, 512, 512, name ="conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = conv_layer(pool4, 512, 512, name ="conv5_1")
    conv5_2 = conv_layer(conv5_1, 512, 512, name ="conv5_2")
    conv5_3 = conv_layer(conv5_2, 512, 512, name ="conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    fc6 = fc_layer(pool5, 25088, 4096, name="fc6")
    assert fc6.get_shape().as_list()[1:] == [4096]
    relu6 = tf.nn.relu(fc6)

    fc7 = fc_layer(relu6, 4096, 4096, name="fc7")
    relu7 = tf.nn.relu(fc7)

    fc8 = fc_layer(relu7, 4096, 1000, name="fc8")
    relu8 = tf.nn.relu(fc8)

    fc9 = fc_layer(relu8, 1000, 8, name="fc8")

    prob = tf.nn.softmax(fc9, name="prob")
    
    cost = tf.reduce_mean(tf.square(tf.subtract(lbl, prob)))

    return {'inputData':inputData, 'classLabels':classLabels, 'prob':prob, 'cost':cost, 'lbl': lbl}
    # data_dict = None
    # print(("build model finished: %ds" % (time.time() - start_time)))

def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
        # filt = get_conv_filter(name)
        filter_size = 3
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        var_name = name + "_filters"
        filt = tf.Variable(initial_value, name=var_name)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        var_name = name + "_biases"
        conv_biases = tf.Variable(initial_value, name=var_name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

def fc_layer(bottom, in_size, out_size, name):

    with tf.variable_scope(name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        var_name = name + "_weights"
        weights = tf.Variable(initial_value, name=var_name)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        var_name = name + "_biases"
        biases = tf.Variable(initial_value, name=var_name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc

def fwrite (fname = 'input.txt', write_data='result'):
        """Write results into the file by Fayeem Aziz"""
        f = open(fname,"w+")
        np.savetxt(fname, write_data, delimiter=',')
        f.close()


path = 'etProjectResult/1/'
if not os.path.exists(path):
        os.makedirs(path)

start_time = time.time()


data = np.genfromtxt('data_resize_exp1.csv', delimiter=',')
data = np.float32(data)
print(data.shape)
# rgb = data[:100,:]

label = np.genfromtxt('labels_exp1.csv', delimiter=',')
label = np.float32(label)
print(label.shape)
label = label[1:,1:]
print(label.shape)
# lbl = label[1:101,1:]

num_batch = 12
dataBatch = np.array_split(data,num_batch)
labelBatch = np.array_split(label,num_batch)


vggMap = vgg16()

cost = vggMap['cost']

n_epochs = 1000
learningRate = 0.001
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

lossArray = []

for epoch_i in range(n_epochs):
    for batch_i in range(num_batch):
        dataBatch_i = dataBatch[batch_i]
        labelBatch_i = labelBatch[batch_i]
        sess.run(optimizer, feed_dict={vggMap['inputData']:dataBatch_i,vggMap['classLabels']:labelBatch_i})
    loss = sess.run(cost, feed_dict={vggMap['inputData']:data,vggMap['classLabels']:label})
    print('Epoch = ', epoch_i, ', Loss = ', loss )
    lossArray.append(loss)

fwrite(os.path.join(path,"loss"),lossArray)

# %-------------------------------------------

prob = sess.run(vggMap['prob'], feed_dict={vggMap['inputData']:data, vggMap['classLabels']:label})
fwrite(os.path.join(path,"probabilities"),prob)
# print(type(prob))
# print(prob.shape)

# labels = sess.run(vggmap['lbl'], feed_dict={vggmap['inputData']:rgb, vggmap['classLabels']:lbl})
# print(type(labels))
# print(labels.shape)

# cost = sess.run(vggmap['cost'], feed_dict={vggmap['inputData']:rgb, vggmap['classLabels']:lbl})
# print(cost)

# print(prob)
# print('---------------')
# print(labels)