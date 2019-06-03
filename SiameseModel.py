import keras
import numpy as np
import matplotlib.pyplot as plt

import random

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from keras.models import load_model
num_classes = 10
epochs = 40
train = True
#train = False

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_rand_batch_pairs(x, digit_indices,batch):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
        
    for d in range(int(batch/2)):
        n = np.random.randint(10)
        ind1 = np.random.randint(len(digit_indices[n]))
        ind2 = np.random.randint(len(digit_indices[n]))
        z1, z2 = digit_indices[n][ind1], digit_indices[n][ind2]
        pairs += [[x[z1], x[z2]]]
        
        dn = np.random.randint(10)
        while dn == n:
            dn = np.random.randint(10)
            
        ind3 = np.random.randint(len(digit_indices[dn]))
        z1, z2 = digit_indices[n][ind1], digit_indices[dn][ind3]
        pairs += [[x[z1], x[z2]]]
        labels += [1, 0]    
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(512, activation='relu')(x)
#    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
train_digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
train_pairs, train_y = create_pairs(x_train, train_digit_indices)

test_digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
test_pairs, test_y = create_pairs(x_test, test_digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
# keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
model.summary()

# train
if train == True:
    # rms = RMSprop()
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    
#    history=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#              batch_size=128,
#              epochs=epochs,verbose=2,
#              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
#    
#
#    plt.figure(figsize=(8, 4))
#    plt.subplot(1, 2, 1)
#    plot_train_history(history, 'loss', 'val_loss')
#    plt.subplot(1, 2, 2)
#    plot_train_history(history, 'accuracy', 'val_accuracy')
#    plt.show()
    

#    # compute final accuracy on training and test sets
#    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
#    tr_acc = compute_accuracy(tr_y, y_pred)
#    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#    te_acc = compute_accuracy(te_y, y_pred)
#    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    start = 0    
    for it in range(50000):
#        train_loss, train_accuracy = model.train_on_batch(
#                [tr_pairs[:, 0], tr_pairs[:, 1]], tr_y)
        train_pairs, train_y = create_rand_batch_pairs(x_train, train_digit_indices,256)
        train_loss, train_accuracy = model.train_on_batch(
                [train_pairs[:, 0], train_pairs[:, 1]], train_y)
 
        if it % 100 == 0:
            print('iteration:',it,'loss:',train_loss,'accuracy:',train_accuracy)
        if (it+1) % 1000 == 0:
            result = []
            for i in range(len(x_test)):
                test_pairs=[]
#                if i%1000 == 0:
#                    print(i)
                for j in range(num_classes):
        #            a = x_test[digit_indices[0][0]]
                    a = x_test[i]
                    b = x_test[test_digit_indices[j][10]]
                    test_pairs += [[a,b]]
                test_pairs = np.array(test_pairs)
                
                result += [model.predict([test_pairs[:, 0], test_pairs[:, 1]])]
                
            result = np.array(result)[:,:,0]
            pre = np.argmin(result,axis=1)
            acc = np.mean(pre==y_test)
            print('test accuracy:',acc)
            
        start += 1
    path = './model/model_'+str(start)+'.h5'
    model.save(path)
    print('model save:',path)

elif train == False:    
    model = load_model('./model/model.h5',custom_objects={'contrastive_loss': contrastive_loss,'accuracy':accuracy})
#    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#    te_acc = compute_accuracy(te_y, y_pred)
    
        
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    #单个数据测试
#    res = []  
#    for i in range(num_classes):
#        a = np.expand_dims(x_test[digit_indices[0][0]],axis = 0)
#        b = np.expand_dims(x_test[digit_indices[i][10]],axis = 0)
#        
#        plt.subplot(1, 2, 1)
#        plt.imshow(a[0,:,:])
#        plt.subplot(1, 2, 2)
#        plt.imshow(b[0,:,:])
#    
#        result = model.predict([a,b])
#        res += [result]
    
    
    #分类数据测试
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    result = []
    num_kind = 6
    for i in range(len(digit_indices[num_kind])):
        test_pairs = []
        if i%1000 == 0:
            print(i)
        for j in range(num_classes):
#            a = x_test[digit_indices[0][0]]
            a = x_test[digit_indices[num_kind][i]]
            b = x_test[digit_indices[j][10]]
            test_pairs += [[a,b]]
        test_pairs = np.array(test_pairs)
    
        result += [model.predict([test_pairs[:, 0], test_pairs[:, 1]])]
    result = np.array(result)[:,:,0]
    pre = np.argmin(result,axis=1)
    acc = np.mean(pre==y_test[digit_indices[num_kind]])
    
    #总体数据测试
#    result = []
#    for i in range(len(x_test)):
#        test_pairs=[]
#        if i%1000 == 0:
#            print(i)
#        for j in range(num_classes):
##            a = x_test[digit_indices[0][0]]
#            a = x_test[i]
#            b = x_test[digit_indices[j][10]]
#            test_pairs += [[a,b]]
#        test_pairs = np.array(test_pairs)
#        
#        result += [model.predict([test_pairs[:, 0], test_pairs[:, 1]])]
#        
#    result = np.array(result)[:,:,0]
#    pre = np.argmin(result,axis=1)
#    acc = np.mean(pre==y_test)