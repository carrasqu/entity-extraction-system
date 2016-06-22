import numpy as np
import sys
import os


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 2 # the 2 comes from the toric code unit cell
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2]) 
            # Convert from [0, 255] -> [0.0, 1.0].
            #images = images.astype(np.float32)
            # images = np.multiply(images, 1.0 / 255.0) # commented since it is ising variables
            #images = np.multiply(images, 1.0 ) # multiply by one, instead
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def extract_labels(nlabels,filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print 'Extracting', filename,'bbbccicicicicib'

    labels=np.loadtxt(filename,dtype='uint8')

    if one_hot:
       print "LABELS ONE HOT"
       print labels.shape
       XXX=dense_to_one_hot(labels,nlabels)
       print XXX.shape
       return dense_to_one_hot(labels,nlabels)
    print "LABELS"
    print labels.shape
    return labels

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print 'Extracting', filename,'aaaaaa'

    #with gzip.open(filename) as bytestream:
    #    magic = _read32(bytestream)
    #    if magic != 2051:
    #        raise ValueError(
    #            'Invalid magic number %d in MNIST image file: %s' %
    #            (magic, filename))
    #    num_images = _read32(bytestream)
    #    rows = _read32(bytestream)
    #    cols = _read32(bytestream)
    #    buf = bytestream.read(rows * cols * num_images)
    #    data = np.frombuffer(buf, dtype=np.uint8)
    #    data = data.reshape(num_images, rows, cols, 1)
    data=np.loadtxt(filename)
    dim=data.shape[0]
    #data=data.reshape(dim,lx,lx,2) # the two comes from the 2 site unite cell of the toric code.
    print data.shape
    return data

def maybe_download(filename, work_directory):
    filepath = os.path.join(work_directory, filename)
    return filepath

def read_data_sets(nlabels, train_dir, fake_data=False, one_hot=False ):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'Xtrain.txt'
    TRAIN_LABELS = 'ytrain.txt'
    TEST_IMAGES = 'Xtest.txt'
    TEST_LABELS = 'ytest.txt'
    #TEST_IMAGES_Trick = 'XtestTrick.txt'
    #TEST_LABELS_Trick = 'ytestTrick.txt'
    VALIDATION_SIZE = 0
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(nlabels,local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(nlabels,local_file, one_hot=one_hot)

    #local_file = maybe_download(TEST_IMAGES_Trick, train_dir)
    #test_images_Trick = extract_images(local_file)
    #local_file = maybe_download(TEST_LABELS_Trick, train_dir)
    #test_labels_Trick = extract_labels(nlabels,local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    #data_sets.test_Trick = DataSet(test_images_Trick, test_labels_Trick)
    return data_sets

def whatis(xx,w2vec):
    digits = set('0123456789') 
    if xx in w2vec:
       return w2vec[xx]
    elif set(xx) & digits:
       return w2vec['DIGIT']
    else: 
       return w2vec['UNK']  
         
def procdata(tset,w2vec,labeldict,wlength,k):
    dat=np.zeros((tset.shape[0],2*wlength+1,k)) 
    lab=np.zeros(tset.shape[0])
    for i in range(tset.shape[0]):
        z=0 
        for j in range(-wlength,wlength+1,1):
             if (i+j<0)|(i+j>tset.shape[0]-1): 
                  # add  PAD
                  dat[i,z,:]=np.transpose(w2vec['PAD'])
                  z+=1    
             elif j==0:
                 # add word i if  and add label
                  dat[i,z,:]= np.transpose( whatis(tset[i][0],w2vec) )
                  lab[i]=labeldict[tset[i][1]]
                  z+=1
             else: 
                  dat[i,z,:]= np.transpose(whatis(tset[i+j][0],w2vec) )   
                  z+=1
    return dat,lab



def datasetup(wlength=3,percentage_train=80,use=0): 

 #reading provided word2vec data 
 w2vec = {}
 with open("wordvecs.txt") as f:
    for line in f:
       key=line.split()[0]
       val=[float(i) for i in line.split()[1:]]
       w2vec[key] = np.asarray(val)

 # expanding the w2vec with word vectors for padding, unknown, and digits
 k=w2vec['asian'].shape[0]
 sigma=np.var(w2vec['asian'])
 mu=np.mean(w2vec['asian'])
 w2vec['PAD']=-np.ones(k)
 np.random.seed(seed=1919191)
 w2vec['UNK']=sigma * np.random.randn(k) + mu # assign a random vector mean mu and variance sigma (vectors in word2vec have approximately similar sigma and mu)
 w2vec['DIGIT']=sigma * np.random.randn(k) + mu

 # loading the training set and storing all the labels in a set the number of labels

 labels = set([])
 with open("news_tagged_data.txt") as f:
    for line in f:
         if line != '\n':
            labels.add(line.split()[1])
 #labels dictionary
 labeldict={}
 for i in range(len(labels)):
  labeldict[list(labels)[i]]=i
 # inverse map
 inv_label_dict = {v: k for k, v in labeldict.items()}


 if use==1: # use=0 generates data sets and writes them to files train, use=1 use in the predictions or if files are generated already
  return labeldict,  inv_label_dict,  w2vec,k
 elif use==0:
     
    
  # test and training set details
  # Context window size
  #wlength=3
  # size of the training set
  #percentage_train=80

  #process the data  query by query
  i=0
  z=0
  
  with open("news_tagged_data.txt") as f:
    for line in f:
         if i==0:
            tset=[line.split()[0],line.split()[1]]
            i+=1
         elif (line.isspace() == False)&(i>0):
            tset=np.vstack((tset,[line.split()[0],line.split()[1]]))
            i+=1     
         elif (line.isspace() == True):
            #print 'empy line',z 
            #print tset.shape 
            pp,labb=procdata(tset,w2vec,labeldict,wlength,k)
            #print pp.shape  
            if z==0 :
             z+=1
             labels=np.reshape(labb,(labb.shape[0],1) )             
             images=pp
             i=0
             #break 
            else:    
             labels=np.vstack((labels, np.reshape(labb,(labb.shape[0],1) )))
             images=np.vstack((images,pp)) 
             i=0
             z+=1 

  
  images=np.reshape(images,(images.shape[0],images.shape[1]*images.shape[2])) # flatten the array to write
 
  itrain=int(percentage_train*labels.shape[0]/100)
   
  np.savetxt('ytrain.txt',labels[0:itrain],fmt='%i')
  np.savetxt('Xtrain.txt',images[0:itrain,:])
  np.savetxt('ytest.txt',labels[itrain:],fmt='%i')
  np.savetxt('Xtest.txt', images[itrain:,:]) 
  return labeldict,  inv_label_dict,w2vec,k
 
         
