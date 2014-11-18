import cPickle
import gzip
import time
import PIL.Image

import random
import numpy
import os

from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images, convertToCMYK
from logistic_sgd import load_data

class RBMachine(object) :
	def __init__(self, input=None, visibleUnits=28*28, hiddenUnits=100, classesUnits=10, Weights=None, CWeights=None, hiddenBias=None, 
				visibleBias=None) :
		numpy_rng = numpy.random.RandomState(1234);
		self.visible = visibleUnits;
		self.hidden = hiddenUnits;
		self.classes = classesUnits;
		self.vbias = numpy.zeros(visibleUnits, dtype='float32');
		self.hbias = numpy.zeros(hiddenUnits, dtype='float32');
		self.cbias = numpy.zeros(classesUnits, dtype='float32');
		self.allC = [];
		self.numpy_rng = numpy_rng;
		self.currentEpoch = 1
		self.expectations = numpy.zeros(hiddenUnits, dtype='float32');
		self.lastHidden = numpy.zeros(hiddenUnits, dtype='float32');

		self.temprature = numpy.zeros(hiddenUnits, dtype='float32');
		self.heatMomentum = numpy.zeros(hiddenUnits, dtype='float32');
		self.coolMomentum = numpy.zeros(hiddenUnits, dtype='float32');

		for x in range(0,classesUnits) :
			self.allC.append(self.convertToArray(x));

		self.W = numpy.asarray(numpy_rng.uniform(low=0,
								 high=4*numpy.sqrt(6. /(visibleUnits + hiddenUnits) ),
								 size=(visibleUnits, hiddenUnits)), dtype='float32');
		print self.W
		self.U = numpy.asarray(numpy_rng.uniform(low=0,
								 high=4*numpy.sqrt(6. / (hiddenUnits + classesUnits) ),
								 size=(classesUnits, hiddenUnits)), dtype='float32')
		self.numpy_rng = numpy_rng;


		self.params = [self.W, self.U, self.cbias, self.vbias, self.hbias];

		self.h = numpy.zeros(hiddenUnits, dtype='float32');
		self.v = numpy.zeros(visibleUnits, dtype='float32');
		self.c = numpy.zeros(classesUnits, dtype='float32');
		
		self.learningRate = .001;

	def setH (self, x) :
		self.h = numpy.array(x);
	def setC (self, x) :
		self.c = numpy.array(x);
	def setV (self, x) :
		self.v = numpy.array(x);
	def getW (self) :
		return self.W
	def units (self) : 
		return [self.v, self.h, self.c]

	def convertToArray(self, bit) :
		ret = numpy.zeros(self.classes, dtype='float32');
		ret[bit] = 1;
		return ret;

	def dropout(self, modification=False) :
		self.expectations = ( self.expectations + self.lastHidden )
		if modification:
			self.computeTemprature();
			dropout = numpy.asarray(self.temprature < self.numpy_rng.uniform(60.,100.,self.hidden)).astype(int)
		else :
			dropout = numpy.asarray(self.numpy_rng.uniform(0.,1.,self.hidden));
			dropout = (dropout > self.numpy_rng.uniform(0.,1.,self.hidden)).astype(int);
		return dropout

	def computeTemprature(self) :
		heat = self.h * self.numpy_rng.uniform(1.,10.,self.hidden)
		cool = numpy.logical_xor(self.h, numpy.ones(self.hidden));
		self.temprature = self.temprature + self.h
		decay = cool * -self.numpy_rng.uniform(0.1,5.,self.hidden)
		self.heatMomentum = (self.heatMomentum * cool) + cool;
		self.coolMomentum = (self.heatMomentum * self.h) + self.h;
		self.temprature = (self.temprature + (numpy.power(heat, self.coolMomentum)))
		self.temprature = (self.temprature + (decay * self.heatMomentum)).clip(0)
	def propUp(self) :
		self.currentEpoch += 1
		sig = self.sigmoid(numpy.dot(self.v, self.W) + numpy.dot(self.c, self.U) + self.hbias);
		unit = (sig > self.numpy_rng.uniform(0.5,1.,self.hidden)).astype(int);
		dropout = self.dropout(modification=False)
		unit = unit * dropout;
		self.lastHidden = unit
		self.setH(unit);
		return self.h;

	def propDown(self) :
		beforeSig = numpy.dot(self.h, self.W.T) + self.vbias;
		beforeSig = (self.sigmoid(beforeSig) > self.numpy_rng.uniform(0.,1.,self.visible)).astype(int);
		self.setV(beforeSig);
		self.p_y_h();
		return [self.c, self.v]

	def p_y_h(self) :

		probabilities = []
		for x in self.allC :
			probabilities.append( numpy.exp(numpy.dot(self.cbias.T, x) + numpy.dot(numpy.dot(self.h,self.U.T),x)) )
		denom = sum(probabilities);

		maxc = self.c;
		maxValue = 0;

		for x, y in zip(self.allC, probabilities) :
			newValue = y/denom
			if (newValue) > maxValue :
				maxValue = (y/denom);
				maxc = x;
		self.setC( maxc );


	def dream(self, x) :
		beforeSig = numpy.dot(numpy.array(x), self.W) + self.hbias;
		beforeSig = (self.sigmoid(beforeSig) > self.numpy_rng.uniform(0.5,1.,self.hidden)).astype(int);
		self.setH(beforeSig);
		return self.propDown();

	def unpickle(self, file) :
	    import cPickle
	    fo = open(file, 'rb')
	    dict = cPickle.load(fo)
	    fo.close()
	    return dict
	
	def testRun(self) :
		#datasets = load_data('mnist.pkl.gz', returnAsIs=True);
		"""cifar = [];
		train_set_x = []
		train_set_y = []
		for x in range(1,6) :
			cifar = self.unpickle('data_batch_'+str(x));
			train_set_x.extend(cifar['data'])
			train_set_y += cifar['labels']
		cifar = self.unpickle('test_batch');
		test_set_x , test_set_y = cifar['data'], cifar['labels']

		
		RGBArray = numpy.zeros((32,32,3),'uint8');
		setx = numpy.asarray(train_set_x[10]);
		R , G, B = numpy.hsplit((setx),3)

		RGBArray[..., 0] = numpy.asarray(R).reshape(32,32);
		RGBArray[..., 1] = numpy.asarray(G).reshape(32,32);
		RGBArray[..., 2] = numpy.asarray(B).reshape(32,32);
		image = PIL.Image.fromarray(RGBArray);
		name = 'samples.png';
		image.save(name);

		dataset = [[0.,0.], [0.,1.], [1.,0.], [1.,1.]]
		classes = [[1.,0.],[0.,1.],[0.,1.],[1.,0.]]"""
		k = 0;
		i = 0;
		correct = 0;
		batchsize = 20

		datasets = load_data('mnist.pkl.gz', returnAsIs=True);
		train_set_x, train_set_y = datasets[0]
		test_set_x, test_set_y = datasets[2]
		for i in range(0,1) :
			self.Gibbs(3,train_set_x[10], self.convertToArray(train_set_y[10]));
		print self.dream(train_set_x[10])[0], self.convertToArray(train_set_y[10]);
		#image = PIL.Image.fromarray(convertToCMYK(numpy.hsplit(self.W.T,3)));
		image = PIL.Image.fromarray(tile_raster_images(
                 X=self.W.T,
                 img_shape=(28, 28), tile_shape=(10, 10),
                 tile_spacing=(1, 1)))
		name = 'filters_idx_iter_'+str(100000)+'.png';
		image.save(name);

		for x in range(0,100) :
			for batchidx in xrange (batchsize) :
				for data, c in zip(train_set_x[batchidx*batchsize:(batchidx+1)*batchsize], train_set_y[batchidx*batchsize:(batchidx+1)*batchsize]) :
					self.Gibbs(2, data, self.convertToArray(c));
		#image = PIL.Image.fromarray(convertToCMYK(numpy.hsplit(self.W.T,3)));
		image = PIL.Image.fromarray(tile_raster_images(
                 X=self.W.T,
                 img_shape=(28, 28), tile_shape=(10, 10),
                 tile_spacing=(1, 1)))
		name = 'filters_idx_iter10k_'+str(100000)+'.png';
		image.save(name);
		correct = 0;
		total = 0;
		for data, c in zip(test_set_x, test_set_y) :
			if numpy.array_equal(self.convertToArray(c),self.dream(data)[0]) :
				correct += 1

		print correct / float(len(test_set_x))
		print correct

	def Gibbs(self, k=1, x=None, y=None) :
		nInput = x;
		yInput = y;
		xnot = 0.
		hnot = 0.
		yone = 0.
		hone = 0.
		for i in range(0,k) :
			self.setV(x)
			self.setC(y)
			nInput = self.v
			yInput = self.c
			hnot = self.propUp();
			yone, xnot = self.propDown();
			hone = self.propUp();
			nInput = xnot
			yInput = yone
		self.updateParameters(numpy.array(x), numpy.array(xnot), numpy.array(y), numpy.array(yone), numpy.array(hnot), numpy.array(hone));
		return nInput;

	def Energy(self) :
		dotproduct = numpy.dot(self.h.T, numpy.dot(self.v, self.W))
		dotproducty = numpy.dot(self.h.T, numpy.dot(self.c, self.U))

		#biases
		hiddenVis = numpy.dot(self.v, self.vbias)
		hiddenClass = numpy.dot(self.c, self.cbias)
		hiddenhid = numpy.dot(self.h, self.hbias)

		return numpy.exp(-(-dotproduct - dotproducty - hiddenVis - hiddenClass - hiddenhid));

	def free_energy(self, x, y) :
		dotproduct = numpy.dot(x, self.W)
		dotproducty = numpy.dot(y, self.U)
		hiddenUnitValue = self.sigmoid(self.hbias + dotproduct);
		hiddenUnitValuenot = self.sigmoid(self.hbias + dotproducty)
		visibleUnitValue = numpy.dot(self.vbias, numpy.array(x).T);
		return numpy.exp(-hiddenUnitValue - visibleUnitValue - dotproducty);

	def updateParameters(self, x, xnot, y, ynot, hnot, hone) :
		self.vbias = self.vbias + self.learningRate * (x - xnot);
		self.hbias = self.hbias + self.learningRate * (hnot - hone);
		self.cbias = self.cbias + self.learningRate * (y - ynot);
		maskOn = True
		if maskOn :
			Mask = (self.numpy_rng.uniform(0,1,size=(self.visible,self.hidden)) > 0.5).astype(int)
			self.W = self.W + (Mask * (self.learningRate * (numpy.outer(hnot, x.T) - numpy.outer(hone, xnot.T)).T))
			Mask = (self.numpy_rng.uniform(0,1,size=(self.classes,self.hidden)) > 0.5).astype(int)
			self.U = self.U + (Mask * (self.learningRate * (numpy.outer(hnot, y.T) - numpy.outer(hone, ynot.T)).T))
		else :
			self.W = self.W + ((self.learningRate * (numpy.outer(hnot, x.T) - numpy.outer(hone, xnot.T)).T))
			self.U = self.U + self.learningRate * (numpy.outer(hnot, y.T) - numpy.outer(hnot, ynot.T)).T


	def sigmoid(self, x) :
		return 1 / (1 + numpy.exp(-x))

#RBM = RBMachine(visibleUnits = 784, hiddenUnits=500,classesUnits=10);
RBM = RBMachine();
RBM.testRun();
