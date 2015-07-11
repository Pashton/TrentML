""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy
import scipy
import theano
import theano.tensor as T
import cPickle
import os
import scipy.stats

name = 'Fnt-sig-200-200-90.cus'


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h, m+h, m-h

def loadObj(name):
    """Reads and returns a single file as a numpy array."""
    __location__ = os.path.realpath(os.getcwd())
    wrFile = open(os.path.join(__location__, name),'rb')
    rf = cPickle.load(wrFile)
    return rf
obj = loadObj(name)
print 'results for : ', name
print 'scores: '
print 'average: ', numpy.mean(numpy.asarray(obj[0]), dtype='float64'), ' std: +-', numpy.std(numpy.asarray(obj[0]), dtype='float64')
#print mean_confidence_interval(numpy.asarray(obj[0]))
activations = numpy.asarray(obj[1])
LayerOne = numpy.asarray(activations[:,0])
LayerTwo = numpy.asarray(activations[:,1])
print 'LayerOne Activations'
print numpy.mean(LayerOne, dtype='float64'), ' std: +-', numpy.std(LayerOne, dtype='float64')
print 'LayerTwo Activations'
print numpy.mean(LayerTwo, dtype='float64'), ' std: +-', numpy.std(LayerTwo, dtype='float64')

print obj[0]




