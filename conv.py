import tensorflow as tf
import readtraining 
import sys
import numpy as np

#lx=16
training=25000
bsize=100
init=0 # 0 start from scratch. 1 restart from saved model

wlength=3 # window size for the word context 
percentage_train=80 # percentage of the whole set that will be used as training set
use=0 # 0 is use writing txt files. use 1 if they are generated already or for predictions 

labeldict, inv_label_dict,w2vec,kdim=readtraining.datasetup(wlength,percentage_train,use) # Generates the training/test set files 
# label dictionary and inverse dictionary. w2vec is the word embedding. kdim is the dimension of the vectors embedding the words.  Generates the training/test set files
# large files are generated because I also like to perform t-sne on the training data and see if something can be already seen from the t-sne representation

numberlabels=len(labeldict)

mnist =readtraining.read_data_sets(numberlabels,'./',one_hot=True) # rereadin the training/test sets

print numberlabels
print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# defining the convolutional and max pool layers
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# defining the model

x = tf.placeholder("float", shape=[None, kdim*(2*wlength+1)]) # placeholder for the spin configurations
#x = tf.placeholder("float", shape=[None, lx*lx*2]) #with padding and no PBC conv net
y_ = tf.placeholder("float", shape=[None, numberlabels])


#first layer 
# convolutional layer # 2*window+1 x 5 patch size, 1 channel (1 "color"), 64 feature maps computed
nmaps1=64
W_conv1 = weight_variable([2*wlength+1, 2, 1,nmaps1])
# bias for each of the feature maps
b_conv1 = bias_variable([nmaps1])

# applying a reshape of the data to get the two dimensional structure back
#x_image = tf.reshape(x, [-1,lx,lx,2]) # #with padding and no PBC conv net
x_image = tf.reshape(x, [-1,2*wlength+1,kdim,1]) # with PBC 

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1) # removing the maxpool layer
h_pool1=h_conv1

#In order to build a deep network, we stack several layers of this type. The second layer will have 8 features for each 5x5 patch. 

# weights and bias of the fully connected (fc) layer. Ihn this case everything looks one dimensiona because it is fully connected
nmaps2=64

#W_fc1 = weight_variable([(lx/2) * (lx/2) * nmaps1,nmaps2 ]) # with maxpool
W_fc1 = weight_variable([(kdim-1) * nmaps1,nmaps2 ]) # no maxpool images remain the same size after conv

b_fc1 = bias_variable([nmaps2])

# first we reshape the outcome h_pool2 to a vector
#h_pool1_flat = tf.reshape(h_pool1, [-1, (lx/2)*(lx/2)*nmaps1]) # with maxpool

h_pool1_flat = tf.reshape(h_pool1, [-1, (kdim-1)*nmaps1]) # no maxpool
# then apply the ReLU with the fully connected weights and biases.
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer. Finally, we add a softmax layer, just like for the one layer softmax regression above.

# weights and bias
W_fc2 = weight_variable([nmaps2, numberlabels])
b_fc2 = bias_variable([numberlabels])

# apply a softmax layer
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#Train and Evaluate the Model
# cost function to minimize

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

if init==0 :
   sess.run(tf.initialize_all_variables())
elif init==1:
   sess.run(tf.initialize_all_variables()) 
   # Add ops to save and restore all the variables.
   saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
   saver.restore(sess, "./model.ckpt")  

for i in range(training):
  batch = mnist.train.next_batch(bsize)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 
    #print "test Trick accuracy %g"%sess.run(accuracy, feed_dict={
    #x: mnist.test_Trick.images, y_: mnist.test_Trick.labels, keep_prob: 1.0})  
#  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})



saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
save_path = saver.save(sess, "./model.ckpt")
print "Model saved in file: ", save_path


