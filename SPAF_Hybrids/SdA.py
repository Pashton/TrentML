"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import os
import sys
import time
import PIL.Image
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images, convertToCMYK
from logistic_sgd import LogisticRegression, load_data, load_faces, load_news, load_cifar, load_caltech101, load_lfw, load_alph
from mlp import HiddenLayer
from dA import dA, gradient_updates_momentum
from Activations import dActivation
import pygal
import cPickle
import matplotlib.pyplot as plt


# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.hybrid_layers = []
        self.dA_layers = []
        self.avActivation = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP
        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
                hlayer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                hlayer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
            hybrid_dA = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=hlayer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          activation='custom')

            self.hybrid_layers.append(hybrid_dA)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          W_prime=sigmoid_layer.W_prime,
                          bhid=sigmoid_layer.b,
                          activation='relu')

            activation_da = dActivation(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          W_prime=sigmoid_layer.W_prime,
                          bhid=sigmoid_layer.b)

            self.dA_layers.append(dA_layer)
            self.avActivation.append(activation_da)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        ncost = 0.
        for x in self.sigmoid_layers :
            ncost = ncost +  T.sum(abs(x.output)) #T.sum(x.output**2)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    def hpretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.hybrid_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    def activation_function(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.avActivation:
            # get the cost and the updates list
            cost = dA.get_activations()
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index
                ],
                outputs=cost,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        #updates = gradient_updates_momentum(self.finetune_cost, self.params, learning_rate, 0.9)
        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def swap_cols(self, a , b, B):
        new = numpy.asarray(a.copy())
        source = b.copy()
        new[:,B] = b[:,B]
        return new
def drawWeights(vh,hh,layer) :
    new = vh.copy()
    new.fill(0)
    for master,k in enumerate(hh.T) :
        #tally up all visible unit connections for this neuron.
        for idx,value in enumerate(k) :
            new[master,:] = new[master,:] + vh[idx,:] * value
    image = PIL.Image.fromarray(tile_raster_images(
        X=numpy.asarray(new),
        img_shape=(28, 28), tile_shape=(20, 10),tile_spacing=(1, 1)))
    image.save(name+'layer-'+str(layer)+'.png')
    return new

def test_SdA(finetune_lr=0.1, pretraining_epochs=10,
             pretrain_lr=0.1, training_epochs=300,
             datasets=None, batch_size=10, arch = [500,900,900], name='pretraining'):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    costs = []
    errors = []
    if datasets == None :
        #datasets = load_data('mnist.pkl.gz')
        datasets = load_alph('alphabet')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print train_set_x.get_value(borrow=True).shape, test_set_x.get_value(borrow=True).shape

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(numpy.random.random_integers(0,500))
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28*28,
        hidden_layers_sizes=arch,
        n_outs=10
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... train hybrid model'
    pretrain_hybrids = sda.hpretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    activation_fns = sda.activation_function(train_set_x=train_set_x,
                                                batch_size=batch_size)
    def pretrainModule(pretrain_fns=None, layer=None, learning_rate=pretrain_lr) :
        print '... pre-training layer ' + str(layer)
        ## Pre-train layer-wise
        corruption_levels = [.3, .3, .3,.3,.3,.3,.3,.3]
        # go through pretraining epochs
        cost_layer = []
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrain_fns[layer](index=batch_index,
                         corruption=corruption_levels[layer],
                         lr=learning_rate))

            #image = PIL.Image.fromarray(tile_raster_images(
            #X=sda.sigmoid_layers[0].W.get_value(borrow=True).T,
            #   img_shape=(28, 28), tile_shape=(20, 10),
            #  tile_spacing=(1, 1)))
            #image = PIL.Image.fromarray(convertToCMYK(numpy.hsplit(sda.sigmoid_layers[0].W.get_value(borrow=True).T,3), (10,10)))
            #image.save(name+'.png')

            #image = PIL.Image.fromarray(tile_raster_images(
            #X=sda.sigmoid_layers[0].W_prime.get_value(borrow=True),
            #    img_shape=(64,64), tile_shape=(10, 10),
            #    tile_spacing=(1, 1)))
            #image.save('filters_corruption_3.png')
            print 'Pre-training layer %i, epoch %d, cost ' % (layer, epoch),
            print numpy.mean(numpy.asarray(c))
            cost_layer.append(numpy.mean(numpy.asarray(c)))
            costs.append(cost_layer)

        #image = PIL.Image.fromarray(convertToCMYK(numpy.hsplit(sda.sigmoid_layers[0].W.get_value(borrow=True).T,3), (10,10)))
    start_time = time.clock()
    pretrainModule(pretraining_fns, 0, pretrain_lr)
    pretrainModule(pretrain_hybrids, 0, pretrain_lr)
    B=numpy.random.randint(200,size=50)
    newWeights = sda.swap_cols(sda.sigmoid_layers[0].W.get_value(borrow=True),
        sda.hybrid_layers[0].W.get_value(borrow=True), B)
    sda.sigmoid_layers[0].W.set_value(newWeights)
    newbias = sda.sigmoid_layers[0].b.get_value(borrow=True).copy()
    newbias[B] = sda.hybrid_layers[0].b.get_value(borrow=True)[B]
    sda.sigmoid_layers[0].b.set_value(newbias)
    pretrainModule(pretraining_fns, 1, 0.01)
    #pretrainModule(pretrain_hybrids, 1)
    #sda.sigmoid_layers[1].W.set_value(newWeights)
    #newWeights = sda.swap_cols(sda.sigmoid_layers[0].W.get_value(borrow=True),
        #sda.hybrid_layers[0].W.get_value(borrow=True), B)
    #sda.sigmoid_layers[1].W.set_value(sda.hybrid_layers[1].W.get_value(borrow=True))
    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################
    learning_rate = theano.shared(numpy.asarray(finetune_lr,
        dtype=theano.config.floatX))

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=learning_rate
    )


    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * 0.989})

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    x, y = [], []
    x_test, y_test = [], []

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                x.append(epoch)
                y.append(this_validation_loss * 100.)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    #test_score = this_validation_loss
                    image = PIL.Image.fromarray(tile_raster_images(
                    X=sda.sigmoid_layers[0].W.get_value(borrow=True).T,
                    img_shape=(28, 28), tile_shape=(20, 10),
                    tile_spacing=(1, 1)))
                    #image = PIL.Image.fromarray(convertToCMYK(numpy.hsplit(sda.sigmoid_layers[0].W.get_value(borrow=True).T,3), (10,10)))
                    image.save('filters_corruption_30.png')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    x_test.append(epoch)
                    y_test.append(test_score * 100.)

                    nextWeights = None
                    for p in range(1,sda.n_layers) :
                        hh = sda.sigmoid_layers[p].W.get_value(borrow=True)
                        if nextWeights is None :
                            nextWeights = drawWeights(sda.sigmoid_layers[0].W.get_value(borrow=True).T,
                                hh,p)
                        else :
                            nextWeights = drawWeights(numpy.asarray(nextWeights), hh, p)
                        if p == (sda.n_layers-1) :
                            hh = sda.logLayer.W.get_value(borrow=True)
                            drawWeights(numpy.asarray(nextWeights), hh, 5)


            if patience <= iter:
                done_looping = True
                break
        #new_learning_rate = decay_learning_rate()
    mean_activation = [];
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(activation_fns[i](index=batch_index))
        #print numpy.mean(numpy.asarray(c), axis=0)
        bar_chart = pygal.Bar()                                            # Then create a bar graph object
        bar_chart.add('Fibonacci', numpy.mean(c,axis=0))  # Add some values
        bar_chart.render_to_file('_chart'+str(i)+'_cus_none.svg')
        mean_activation.append(numpy.mean(c))
        #sda.sigmoid_layers[1].W.set_value(sda.sigmoid_layers[0].W_prime.get_value(borrow=False).T)
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
            ' with mean activation: '
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print "architecture: ", arch
    return [test_score * 100., numpy.asarray([numpy.asarray(x),numpy.asarray(y)]), numpy.asarray([numpy.asarray(mean_activation[0]), numpy.asarray(mean_activation[1])]), ((end_time-start_time)/60.), numpy.asarray([numpy.asarray(x_test),numpy.asarray(y_test)])]

def createGraph(values, title, xlabel, ylabel) :
    fig = plt.figure()
    fig.suptitle(title)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for value in numpy.asarray(values):
        ax.scatter(value[0],value[1], color='blue')
        ax.plot(value[0], value[1], color='blue', linestyle='dashdot')
    plt.show()


if __name__ == '__main__':
    scores = []
    activations = []
    epochs = []
    test_epochs = []
    seeds = [42,50,60,100,200,300,1200,20,19,30,12,72,369,189,39,48,53,64,113,213,239,678,37,89,73,64,5,233,14,320]
    tm = []
    n = 5
    name= 'MNIST-test-'
    datasets = load_data('mnist.pkl.gz')
    for x in xrange(n) :
        #datasets = load_alph('Fnt', seed=x)
        print name, x
        name = name + str(x)
        network = test_SdA(datasets= datasets, arch = [200,200],name= name+str(x))
        scores.append(network[0])
        act = network[2]
        activations.append([numpy.mean(act[0]), numpy.mean(act[1])])
        epochs.append(network[1])
        test_epochs.append(network[4])
        tm.append(network[3])
    print 'average: ', numpy.mean(numpy.asarray(scores), dtype='float64'), ' std: +-', numpy.std(numpy.asarray(scores), dtype='float64')
    print 'average activations Layer 1: '
    actv = numpy.asarray(activations)
    print 'average: ', numpy.mean(actv[:,0], dtype='float64'), ' std: +-', numpy.std(actv[:,0], dtype='float64')
    print 'average activations Layer 2: '
    print 'average: ', numpy.mean(actv[:,1], dtype='float64'), ' std: +-', numpy.std(actv[:,1], dtype='float64')
    print name
    filename = name + '.cus'
    __location__ = os.path.realpath(os.getcwd())
    wrFile = open(os.path.join(__location__, filename),'wb')
    cPickle.dump([numpy.asarray(scores),activations,epochs,test_epochs,numpy.asarray(tm)],wrFile)
    wrFile.close()
    createGraph(epochs, 'MNIST Validation Error Per Epoch', 'Epoch', 'Error Percent')
    createGraph(test_epochs, 'MNIST Test Error Per Epoch', 'Epoch', 'Error Percent')
