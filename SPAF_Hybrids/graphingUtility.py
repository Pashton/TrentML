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
    __location__ = os.path.realpath(os.getcwd())
    wrFile = open(os.path.join(__location__, 'MNIST-200-200-SIGMOIDAL.cus'),'rb')
    network = cPickle.load(wrFile)
    print network
    createGraph(epochs, 'MNIST Validation Error Per Epoch', 'Epoch', 'Error Percent')
    createGraph(test_epochs, 'MNIST Test Error Per Epoch', 'Epoch', 'Error Percent')
    wrFile.close()
