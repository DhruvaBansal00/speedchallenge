import tensorflow as tf
import numpy as np
print(tf.__version__)

learning_rate = 0.001

b = tf.Variable([.3], tf.float32)
W = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

X_train      = [4.0, 0.0, 12.0]
Y_train      = [5.0, 9, -3]
linear_model = W*x + b   # y = W*x + b; 5= -1*4 + 9; 9=1*0 + 9;  -3 = -1*12 + 9

model_delta = tf.square(linear_model - y)
loss        = tf.reduce_sum(model_delta)
optimizer   = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init        = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        feed_dict_batch = {x: X_train, y: Y_train}
        sess.run(optimizer, feed_dict=feed_dict_batch)
    W_value, b_value = sess.run([W, b])
    print(W_value)
    print(b_value)
