# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf
import math

def weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights


    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

    return weights


def biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases

    bias = tf.Variable(tf.zeros(n_labels))

    return bias
    


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    
    y=tf.add(tf.matmul(input, w), b)

    return y


def softmax(x):
    """Compute softmax values for each sets of scores in x.""" 
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax =tf.nn.softmax(logits)     
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output


softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)


D=tf.reduce_sum(-tf.mul(one_hot, tf.log(softmax)))

# TODO: Print cross entropy from session
with tf.Session() as sess:
    # TODO: Feed the x tensor 123
    output = sess.run(D, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)




def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    
    result=list()
    
    n_batches=int(math.floor(len(features)/batch_size))
    last_batch_size=int((len(features)-n_batches*batch_size))
    
    
    for i in range(n_batches):
        this_batch_features=list()
        this_batch_labels=list()
        for j in range(batch_size):
            this_batch_features.append(features[i*batch_size+j])
            this_batch_labels.append(labels[i*batch_size+j])
        result.append([this_batch_features, this_batch_labels])
    
    last_batch_features=list()
    last_batch_labels=list()
    for j in range(last_batch_size):
        last_batch_features.append(features[n_batches*batch_size+j])
        last_batch_labels.append(labels[n_batches*batch_size+j])
    result.append([last_batch_features, last_batch_labels])
    
    
    return result

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    outout_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
        
    return outout_batches





