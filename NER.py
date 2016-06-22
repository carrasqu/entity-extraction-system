import tensorflow as tf
import readtraining
import sys
import numpy as np

def whatis(xx,w2vec):
    digits = set('0123456789')
    if xx in w2vec:
       return w2vec[xx]
    elif set(xx) & digits:
       return w2vec['DIGIT']
    else:
       return w2vec['UNK']

def procdataNER(tset,w2vec,wlength,k):
    dat=np.zeros((tset.shape[0],2*wlength+1,k))
    for i in range(tset.shape[0]):
        z=0
        for j in range(-wlength,wlength+1,1):
             if (i+j<0)|(i+j>tset.shape[0]-1):
                  # add  PAD
                  dat[i,z,:]=np.transpose(w2vec['PAD'])
                  z+=1
             elif j==0:
                 # add word i if  and add label
                  dat[i,z,:]= np.transpose( whatis(tset[i],w2vec) )
                  z+=1
             else:
                  dat[i,z,:]= np.transpose(whatis(tset[i+j],w2vec) )
                  z+=1
    return dat

def pquery(s,wlength,k,w2vec):
  #process the query to make it look like the input of our model
  images=np.zeros((1,2*wlength+1,k))
  s=np.asarray(s.split())
  images=procdataNER(s,w2vec,wlength,k)
  images=np.reshape(images,(images.shape[0],images.shape[1]*images.shape[2])) # flatten the array to make it look like the input of the conv net
  return images




# Do stuff here 

wlength=3
percentage_train=80
use=1 # 0 is use writing txt files. use 1 if they are generated already or for predictions

labeldict, inv_label_dict,w2vec,kdim=readtraining.datasetup(wlength,percentage_train,use) # Generates the training/test set files
# label dictionary and inverse dictionary. w2vec is the word embedding. kdim is the dimension of the vectors embedding the words.  Generates the training/test set files
numberlabels=len(labeldict)

# model definition and restore the conv net  model

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

x = tf.placeholder("float", shape=[None, kdim*(2*wlength+1)]) # placeholder for the word configurations
y_ = tf.placeholder("float", shape=[None, numberlabels]) # placeholder for the labels

#first layer 
# convolutional layer # 2*window+1 x 2 patch size, 1 channel (1 "color"), 64 feature maps computed
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
#sess.run(tf.initialize_all_variables())

# Add ops to save and restore all the variables.
saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
saver.restore(sess, "./model.ckpt")
#print("Model restored.")

prediction=tf.argmax(y_conv,1)

while True :
  s=raw_input('Type a query (type "exit" to exit): \n')
  print " "  
  if s=='exit':
     break
  else:
     images=pquery(s,wlength,kdim,w2vec)
     predictions=sess.run(prediction,feed_dict={ x:images, keep_prob: 1.0})
     z=0
     for i in s.split(): 
         print i, inv_label_dict[predictions[z]]
         z+=1
     np.savetxt('checking.txt', images)
     

