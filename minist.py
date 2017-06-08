from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28), n 
n_classes = 10  # MNIST total classes (0-9 digits), k

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input]) #1 x n
labels = tf.placeholder(tf.float32, [None, n_classes]) #1 x k

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes])) #n x k
bias = tf.Variable(tf.random_normal([n_classes])) #k x 1

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias) #1xnxnxk + 1xk= 1xk

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) #cost=mean(cross_entropy(softmax(logits), labels)) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) #GradientDescentOptimizer runs SGD?

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)) #if they are equal then set as 1, e.g [1, 1, 1, 0, 0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #sum them up and mean 


# TODO: Set batch size
batch_size = 128
assert batch_size is not None, 'You must set the batch size'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # TODO: Train optimizer on all batches, cut into mini batches for computational efficiency
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels}) #logits<-test_features,labels<-test_labels, 
    #doesn't need weights(already optimized) 

print('Test Accuracy: {}'.format(test_accuracy))
