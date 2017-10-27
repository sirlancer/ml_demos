import tensorflow as tf
import numpy as np

############## low-level TensorFlow library ############
# Model parameters
W = tf.Variable(np.array([.3]), dtype=tf.float32)
b = tf.Variable(np.array([-.3]), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# initialize W and b
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# training data
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print('W:%s, b:%s, loss:%s' % (curr_W, curr_b, curr_loss))

####################### high-level TensorFlow library ######################
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics:%r" % train_metrics)
print("eval metrics:%r" % eval_metrics)
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 0.1 + 0.3
#
# Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# bias = tf.Variable(tf.zeros([1]))
# y = Weights * x_data + bias
#
# loss = tf.reduce_mean((y_data - y) ** 2)
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for i in range(200):
#     sess.run(train)
#     if i % 20 == 0:
#         print(i,sess.run(Weights), sess.run(bias), sess.run(loss))