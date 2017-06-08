import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([ [123,456,789] ]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

tensors=list()

tensors.append(A)
tensors.append(B)
tensors.append(C)


with tf.Session() as sess:
    # Run the tf.constant operation in the session
    for x in tensors:
    	output = sess.run(x)
    	print(output)

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
x=tf.constant(10)
y=tf.constant(2)
z=tf.sub(tf.div(x, y), 1)

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
print(weights)

bias = tf.Variable(tf.zeros(n_labels))
print(bias)

with tf.Session() as sess:
    # TODO: Feed the x tensor 123
    output = sess.run(z)
    print(output)