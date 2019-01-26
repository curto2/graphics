
# Original python notebook from Fridman <fridman@mit.edu>
# Slightly modified by Curtó and Zarza
# {curto,zarza}@estudiants.urv.cat

#!/usr/bin/python2

import sys, os, time
import itertools
import math, random
import glob
import tensorflow as tw
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Basic parameters

m_epochs = 22
path_class = "gender/" # Adapt for specific class, generate folder with subfolders for each attribute
attributes = ["male","female"] # Add specific attributes
sample_x = 32
sample_y = 32
train_test_split_ratio = 0.9
batch_size = 32
checkpoint_name = "output/cz.ckpt"

# Helper layer functions

def weight_variable(shape):
    initial = tw.truncated_normal(shape, stddev=0.1)
    return tw.Variable(initial)

def bias_variable(shape):
    initial = tw.constant(0.1, shape=shape)
    return tw.Variable(initial)

def ctn2d(x, W, stride):
    return tw.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def m_pool_2x2(x):
    return tw.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Model

x = tw.placeholder(tw.float32, shape=[None, sample_x, sample_y, 3])
y_ = tw.placeholder(tw.float32, shape=[None, len(attributes)])

x_2 = x

# Our first six convolutional layers of 16 3x3 filters

W_ctn1 = weight_variable([3, 3, 3, 16])
b_ctn1 = bias_variable([16])
h_ctn1 = tw.nn.relu(ctn2d(x_2, W_ctn1, 1) + b_ctn1)

W_ctn2 = weight_variable([3, 3, 16, 16])
b_ctn2 = bias_variable([16])
h_ctn2 = tw.nn.relu(ctn2d(h_ctn1, W_ctn2, 1) + b_ctn2)

W_ctn3 = weight_variable([3, 3, 16, 16])
b_ctn3 = bias_variable([16])
h_ctn3 = tw.nn.relu(ctn2d(h_ctn2, W_ctn3, 1) + b_ctn3)

W_ctn4 = weight_variable([3, 3, 16, 16])
b_ctn4 = bias_variable([16])
h_ctn4 = tw.nn.relu(ctn2d(h_ctn3, W_ctn4, 1) + b_ctn4)

W_ctn5 = weight_variable([3, 3, 16, 16])
b_ctn5 = bias_variable([16])
h_ctn5 = tw.nn.relu(ctn2d(h_ctn4, W_ctn5, 1) + b_ctn5)

W_ctn6 = weight_variable([3, 3, 16, 16])
b_ctn6 = bias_variable([16])
h_ctn6 = tw.nn.relu(ctn2d(h_ctn5, W_ctn6, 1) + b_ctn6)

# Our pooling layer

h_pool4 = m_pool_2x2(h_ctn6)

n1, n2, n3, n4 = h_pool4.get_shape().as_list()

W_fc1 = weight_variable([n2*n3*n4, 2])
b_fc1 = bias_variable([2])

# We flatten our pool layer into a fully connected layer

h_pool4_flat = tw.reshape(h_pool4, [-1, n2*n3*n4])

y = tw.matmul(h_pool4_flat, W_fc1) + b_fc1

sn = tw.InteractiveSession()

# Our loss function and optimizer

loss = tw.reduce_mean(tw.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
train_step = tw.train.AdamOptimizer(1e-4).minimize(loss)
sn.run(tw.initialize_all_variables())

saver = tw.train.Saver()
time_start = time.time()

v_loss = least_loss = 99999999

# Load data

full_set = []

for abts in attributes:
    for ex in glob.glob(os.path.join(path_class, abts, "*")):
        sample = cv2.imread(ex)
        if not sample is None:
            sample = cv2.resize(sample, (32, 32))

            # Create an array representing our classes and set it
            one_hot_array = [0] * len(attributes)
            one_hot_array[attributes.index(abts)] = 1
            assert(sample.shape == (32, 32, 3))

            full_set.append((sample, one_hot_array, ex))

random.shuffle(full_set)

# We split our data into a training and test set here

split_index = int(math.floor(len(full_set) * train_test_split_ratio))
train_set = full_set[:split_index]
test_set = full_set[split_index:]

# We ensure that our training and test sets are a multiple of batch size

train_set_offset = len(train_set) % batch_size
test_set_offset = len(test_set) % batch_size
train_set = train_set[: len(train_set) - train_set_offset]
test_set = test_set[: len(test_set) - test_set_offset]

train_x, train_y, train_z = zip(*train_set)
test_x, test_y, test_z = zip(*test_set)

print("Starting training... [{} training examples]".format(len(train_x)))

v_loss = 9999999
train_loss = []
vn_loss = []

for z in range(0, m_epochs):

    # Iterate over our training set

    for tt in range(0, (len(train_x) // batch_size)):
        start_batch = batch_size * tt
        end_batch = batch_size * (tt + 1)
        train_step.run(feed_dict={x: train_x[start_batch:end_batch], y_: train_y[start_batch:end_batch]})
        seen = "Current epoch, examples seen: {:20} / {} \r".format(tt * batch_size, len(train_x))
        sys.stdout.write(seen.format(tt * batch_size))
        sys.stdout.flush()

    seen = "Current epoch, examples seen: {:20} / {} \r".format((tt + 1) * batch_size, len(train_x))
    sys.stdout.write(seen.format(tt * batch_size))
    sys.stdout.flush()

    t_loss = loss.eval(feed_dict={x: train_x, y_: train_y})
    v_loss = loss.eval(feed_dict={x: test_x, y_: test_y})
    
    train_loss.append(t_loss)
    vn_loss.append(v_loss)

    sys.stdout.write("Epoch {:5}: loss: {:15.10f}, validation loss: {:15.10f}".format(z + 1, t_loss, v_loss))

    if v_loss < least_loss:
        sys.stdout.write(", saving new best model to {}".format(checkpoint_name))
        least_loss = v_loss
        filename = saver.save(sn, checkpoint_name)

    sys.stdout.write("\n")

plt.figure()
plt.xticks(np.arange(0, len(train_loss), 1.0))
plt.ylabel("Loss")
plt.xlabel("Epochs")
train_line = plt.plot(range(0, len(train_loss)), train_loss, 'r', label="Train loss")
vn_line = plt.plot(range(0, len(vn_loss)), vn_loss, 'g', label="Validation loss")
plt.legend()
plt.show()

zipped_x_y = list(zip(test_x, test_y))
cfn_true = []
cfn_ptd = []
for tt in range(0, len(zipped_x_y)):
    q = zipped_x_y[tt]
    sfmax = list(sn.run(tw.nn.softmax(y.eval(feed_dict={x: [q[0]]})))[0])
    sf_idx = sfmax.index(max(sfmax))
    
    predicted_label = attributes[sf_idx]
    actual_label = attributes[q[1].index(max(q[1]))]
    
    cfn_true.append(actual_label)
    cfn_ptd.append(predicted_label)
    
    if predicted_label != actual_label:
        print("Actual: {}, predicted: {}".format(actual_label, predicted_label))
        path_sample = test_z[tt]    
        epl_sample = cv2.imread(filename=path_sample)

# From sklearn docs

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm2 = np.around(cm2, 2)

    threshd = cm.max() / 2.
    for c, z in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(z, c, str(cm[c, z]) + " / " + str(cm2[c, z]),
                 horizontalalignment="center",
                 color="white" if cm[c, z] > threshd else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(cfn_true, cfn_ptd)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=attributes, normalize=False,
                      title='Normalized confusion matrix')
plt.show()

