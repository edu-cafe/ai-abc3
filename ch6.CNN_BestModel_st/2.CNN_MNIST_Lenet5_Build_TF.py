import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DATA_DIR = './data-mnist'

def multi_layers_sparse(train_set, test_set):
    x_train, y_train = train_set.images, train_set.labels
    x_test, y_test = test_set.images, test_set.labels

    #------------------------------------------------------

    # [5, 5, 1, 6] : filter(5,5), ch(1), filter_no(6)
    w1 = tf.get_variable('w1', shape=[5, 5, 1, 6],   # feature(input), class(output)
                         initializer=tf.glorot_uniform_initializer)  # xavier initialization
    w2 = tf.get_variable('w2', shape=[5, 5, 6, 16],
                         initializer=tf.glorot_uniform_initializer)
    w3 = tf.get_variable('w3', shape=[400, 120],
                         initializer=tf.glorot_uniform_initializer)
    w4 = tf.get_variable('w4', shape=[120, 84],
                         initializer=tf.glorot_uniform_initializer)
    w5 = tf.get_variable('w5', shape=[84, 10],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([6]))
    b2 = tf.Variable(tf.zeros([16]))
    b3 = tf.Variable(tf.zeros([120]))
    b4 = tf.Variable(tf.zeros([84]))
    b5 = tf.Variable(tf.zeros([10]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)

    #------------------------------------------------------

    # out:(?, 28, 28, 6), same or valid
    # stride크기 [1,1,1,1]:NHWC(batch_size, Height, Width, Channel)
    c1 = tf.nn.conv2d(ph_x, w1, [1, 1, 1, 1], 'SAME')
    # c1 = tf.nn.conv2d(ph_x, w1, [1, 1, 1, 1], 'VALID') # error!! --> 여기서는 28x28x1 input
    r1 = tf.nn.relu(c1 + b1)
    # [1, 2, 2, 1], [1, 2, 2, 1] : filter & stride
    # out:(?, 14, 14, 6)
    p1 = tf.nn.max_pool2d(r1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    print(c1.shape)  # (?, ?, ?, 6)
    print(r1.shape)  # (?, ?, ?, 6)
    print(p1.shape)  # (?, ?, ?, 6)

    # out:(?, 10, 10, 16), same or valid
    c2 = tf.nn.conv2d(p1, w2, [1, 1, 1, 1], 'VALID')
    # c2 = tf.nn.conv2d(p1, w2, [1, 1, 1, 1])
    r2 = tf.nn.relu(c2 + b2)
    # out:(?, 5, 5, 16)
    p2 = tf.nn.max_pool2d(r2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    # p2 = tf.nn.max_pool2d(r2, [1, 2, 2, 1], [1, 2, 2, 1])
    print(c2.shape)  # (?, ?, ?, 6)
    print(r2.shape)  # (?, ?, ?, 16)
    print(p2.shape)  # (?, ?, ?, 16)

    flat = tf.reshape(p2, (-1, 5 * 5 * 16))
    print(flat.shape)  # (?, 400)
    # exit(-1)


    z3 = tf.matmul(flat, w3) + b3
    r3 = tf.nn.sigmoid(z3)
    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.sigmoid(z4)
    z = tf.matmul(r4, w5) + b5

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z,
                                                        labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)  #
    # optimizer = tf.train.GradientDescentOptimizer(0.01)  #
    # optimizer = tf.train.AdamOptimizer(0.01)  #
    optimizer = tf.train.RMSPropOptimizer(0.001)  #
    train = optimizer.minimize(loss)
    #------------------------------------------------------

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 128
    n_iteration = len(x_train )// batch_size

    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            # n1 = j * batch_size
            # n2 = n1 + batch_size
            #
            # xx = x_train[n1:n2]
            # yy = y_train[n1:n2]

            xx, yy = train_set.next_batch(batch_size)  # epoch 마다 shuffle하는 기능이 내장됨!!

            xx = xx.reshape(-1, 28, 28, 1)

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})
            # break

        # break
        print(i, total / n_iteration)
    print('-' * 50)

    xx = x_test.reshape(-1, 28, 28, 1)
    preds = sess.run(z, {ph_x: xx})
    # print(preds)
    pred_arg = np.argmax(preds, axis=1)
    # print(pred_arg)

    print('acc:', np.mean(pred_arg == y_test))

    sess.close()


#mnist = input_data.read_data_sets('mnist') # 6ok train-sets, 10k test-sets
mnist = input_data.read_data_sets(DATA_DIR) # 6ok train-sets, 10k test-sets
#mnist = input_data.read_data_sets(DATA_DIR, one_hot=True) # 6ok train-sets, 10k test-sets
multi_layers_sparse(mnist.train, mnist.test)



# print(mnist.train.images.shape)       # (55000, 784)   784:28*28
# print(mnist.validation.images.shape)  # (5000, 784)
# print(mnist.test.images.shape)        # (10000, 784)
# print(mnist.train.labels.shape)       # (55000, 10)
