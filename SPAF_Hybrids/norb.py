#------------------------------------------
#  read norb image into py-colmajor format
#  Author: Li Wan (ntuwanli@gmail.com)
#------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import struct
import cPickle
import sys
import os
from PIL import Image

import re
import gzip
import zipfile
class UnpickleError(Exception):
	pass
try:
	import magic
	ms = magic.open(magic.MAGIC_NONE)
	ms.load()
except ImportError: # no magic module
	ms = None

DRAW_IMAGE_LABEL = True
IMAGE_TARGET_SIZE = 48
#IMAGE_TARGET_SIZE = 50
IMAGE_SIZE = 108
LABEL_NAME = [ 'animal', 'human', 'plane', 'truck', 'car', 'blank' ]
NUM_IMAGE_BATCH = 29160 
#NUM_IMAGE_BATCH = 30

def pickle(filename, data, compress=False):
	if compress:
		fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
		fo.writestr('data', cPickle.dumps(data, -1))
	else:
		fo = open(filename, "wb")
		cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
	fo.close()
	
def unpickle(filename):
	if not os.path.exists(filename):
		raise UnpickleError("Path '%s' does not exist." % filename)
	if ms is not None and ms.file(filename).startswith('gzip'):
		fo = gzip.open(filename, 'rb')
		dict = cPickle.load(fo)
	elif ms is not None and ms.file(filename).startswith('Zip'):
		fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
		dict = cPickle.loads(fo.read('data'))
	else:
		fo = open(filename, 'rb')
		dict = cPickle.load(fo)
	
	fo.close()
	return dict

def show_norb_image( image_matrix,  title_string ):
	image_size = IMAGE_TARGET_SIZE
	image_matrix_copy = np.round( image_matrix.reshape( (image_size*2,image_size) ).copy() ).astype(np.uint8)
	#import pdb; pdb.set_trace()
	# change order of two image, v order to h order
	image_matrix_copy = np.hstack( 
			(image_matrix_copy[:image_size,:], image_matrix_copy[image_size:,:] ))
	(c,r) = image_matrix_copy.shape

	# plot image
	im = np.zeros( (3,c,r), np.uint8 )
	im[0,:,:] = image_matrix_copy
	im[1,:,:] = image_matrix_copy
	im[2,:,:] = image_matrix_copy
	im_show = im.transpose( (1,2,0) )
	plt.imshow( im_show )
	plt.title( title_string )
	plt.show()

def resize_image( im_data ):
	im_data = im_data.reshape( (IMAGE_SIZE*2, IMAGE_SIZE ) )
	#import pdb; pdb.set_trace()
	# read as PIL image
	im = Image.fromarray( np.uint8(im_data), 'L' )
	# resize image to target size
	im = im.resize( (IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE*2), Image.BICUBIC )
	return np.asarray( im ).reshape( 2*IMAGE_TARGET_SIZE**2 )

def build_batch( raw_data_path, batch_index, start_index, data_type ):
	#----------------------------------------
	#        print current info
	#----------------------------------------
	assert( data_type == 'training' or data_type == 'testing' )
	if data_type == 'training':
		pre_fix = raw_data_path + '/' + 'norb-5x46789x9x18x6x2x108x108-' + data_type + '-'
	else:
		pre_fix = raw_data_path + '/' + 'norb-5x01235x9x18x6x2x108x108-' + data_type + '-'
	#batch_index = 10
	file_name = pre_fix + '%02d' % batch_index + "-"
	print '\n' + file_name
	#---------------------------------------
	#            read images
	#---------------------------------------
	data_file = open( file_name + 'dat.mat','r')
	# skip header
	data_file.read(8)
	num_image_info = data_file.read(4)
	#assert( ord(num_image_info[1]) * 256 + ord(num_image_info[0]) == num_image )
	data_file.read(12)
	# alloc memory for result data
	result_data = np.zeros( (2*(IMAGE_TARGET_SIZE)**2, NUM_IMAGE_BATCH), np.uint8, order='C' )
	# read image data
	for ii in range(NUM_IMAGE_BATCH):
		im_data = data_file.read( IMAGE_SIZE * IMAGE_SIZE * 2)
		im_data = [ ord(e) for e in im_data ] # list of unit8
		im_data = np.array( im_data )
		#import pdb; pdb.set_trace()
		result_data[:,ii] = resize_image( im_data )
		if ii % 100 == 0: 
			print "\r %d/%d" % (ii,NUM_IMAGE_BATCH),
			sys.stdout.flush()

	data_file.close()
	#---------------------------------------
	#            read labels
	#---------------------------------------
	label_file = open( file_name + 'cat.mat', 'r' )
	# skip header
	label_file.read(8);
	num_label_info = label_file.read(4)
	#assert( ord(num_label_info[1]) * 256 + ord(num_label_info[0]) == NUM_IMAGE_BATCH )
	label_file.read(8)
	result_label = []
	for ii in range( NUM_IMAGE_BATCH ):
		result_label.append( struct.unpack('i', label_file.read(4) )[0] )
	label_file.close()

	#---------------------------------------
	#      display image with labels
	#---------------------------------------
	if DRAW_IMAGE_LABEL:
		#import pdb; pdb.set_trace()
		#for ii in range( NUM_IMAGE_BATCH ):
		#    show_norb_image( result_data[:,ii], str(ii) + ": " + 
		#            LABEL_NAME[result_label[ii]] );
		ii = 0
		show_norb_image( result_data[:,ii], str(ii) + ": " + 
				LABEL_NAME[result_label[ii]] );

	#----------------------------------------
	#   store in to dict and dump into file
	#---------------------------------------
	out = {}
	out['data'] = result_data
	out['labels'] = result_label
	out['file-name'] = file_name
	#import pdb; pdb.set_trace()
	pickle( "data_batch_" + str(start_index + batch_index), out )

	return np.mean( result_data, 1 )

def build_batches_meta( image_mean_list ):
	output_file = 'batches.meta'
	out = {}
	out['num_vis' ] = 2*IMAGE_TARGET_SIZE**2 
	out['label_names'] = LABEL_NAME
	out['num_cases_per_batch'] = NUM_IMAGE_BATCH
	# compuate image_mean
	image_mean = image_mean_list[0]
	num_batches = len( image_mean_list )
	for ii in range( num_batches ):
		image_mean += image_mean_list[ii]
	image_mean /= num_batches
	if DRAW_IMAGE_LABEL:
		show_norb_image( image_mean, 'overall mean' )
	out['data_mean'] = image_mean
	pickle( output_file, out )

def main():
	raw_data_path = os.getcwd()
	image_mean_list = []
	# training data
	for ii in range(1,11):
	#for ii in range(1,):
		mean_ii = build_batch( raw_data_path, ii, 0 , 'training' )
		image_mean_list.append( mean_ii )

	# meta file
	#import pdb; pdb.set_trace()
	build_batches_meta( image_mean_list ) 

	# testing data
	for ii in range(1,3):
		build_batch( raw_data_path, ii, 10 , 'testing' )

if __name__ == "__main__":
	#DRAW_IMAGE_LABEL = True
	main()