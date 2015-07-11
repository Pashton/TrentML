""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import bz2
import gzip
import logging
import os
import warnings
try:
    from exceptions import DeprecationWarning
except ImportError:
    pass

import numpy

def loadNORB(cls, which_set, filetype, subtensor):
    """Reads and returns a single file as a numpy array."""

    assert which_set in ['train', 'test']
    assert filetype in ['dat', 'cat', 'info']

    def getPath(which_set):
        dirname = os.path.join(os.getenv('~'),
                               '/datasets')
        if which_set == 'train':
            instance_list = '46789'
        elif which_set == 'test':
            instance_list = '01235'

        filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
            (instance_list, which_set + 'ing', filetype)

        return filename

    def parseNORBFile(file_handle, subtensor=None, debug=False):
        """
        Load all or part of file 'file_handle' into a numpy ndarray
        .. todo::
            WRITEME properly
        :param file_handle: file from which to read file can be opended
          with open(), gzip.open() and bz2.BZ2File()
          @type file_handle: file-like object. Can be a gzip open file.
        :param subtensor: If subtensor is not None, it should be like the
          argument to numpy.ndarray.__getitem__.  The following two
          expressions should return equivalent ndarray objects, but the one
          on the left may be faster and more memory efficient if the
          underlying file f is big.
          read(file_handle, subtensor) <===> read(file_handle)[*subtensor]
          Support for subtensors is currently spotty, so check the code to
          see if your particular type of subtensor is supported.
          """

        def readNums(file_handle, num_type, count):
            """
            Reads 4 bytes from file, returns it as a 32-bit integer.
            """
            num_bytes = count * numpy.dtype(num_type).itemsize
            string = file_handle.read(num_bytes)
            return numpy.fromstring(string, dtype=num_type)

        def readHeader(file_handle, debug=False, from_gzip=None):
            """
            .. todo::
                WRITEME properly
            :param file_handle: an open file handle.
            :type file_handle: a file or gzip.GzipFile object
            :param from_gzip: bool or None
            :type from_gzip: if None determine the type of file handle.
            :returns: data type, element size, rank, shape, size
            """

            if from_gzip is None:
                from_gzip = isinstance(file_handle,
                                       (gzip.GzipFile, bz2.BZ2File))

            key_to_type = {0x1E3D4C51: ('float32', 4),
                           # what is a packed matrix?
                           # 0x1E3D4C52: ('packed matrix', 0),
                           0x1E3D4C53: ('float64', 8),
                           0x1E3D4C54: ('int32', 4),
                           0x1E3D4C55: ('uint8', 1),
                           0x1E3D4C56: ('int16', 2)}

            type_key = readNums(file_handle, 'int32', 1)[0]
            elem_type, elem_size = key_to_type[type_key]
            if debug:
                logger.debug("header's type key, type, type size: "
                             "{0} {1} {2}".format(type_key, elem_type,
                                                  elem_size))
            if elem_type == 'packed matrix':
                raise NotImplementedError('packed matrix not supported')

            num_dims = readNums(file_handle, 'int32', 1)[0]
            if debug:
                logger.debug('# of dimensions, according to header: '
                             '{0}'.format(num_dims))

            if from_gzip:
                shape = readNums(file_handle,
                                 'int32',
                                 max(num_dims, 3))[:num_dims]
            else:
                shape = numpy.fromfile(file_handle,
                                       dtype='int32',
                                       count=max(num_dims, 3))[:num_dims]

            if debug:
                logger.debug('Tensor shape, as listed in header: '
                             '{0}'.format(shape))

            return elem_type, elem_size, shape

        elem_type, elem_size, shape = readHeader(file_handle, debug)
        beginning = file_handle.tell()

        num_elems = numpy.prod(shape)

        result = None
        if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
            assert subtensor is None, \
                "Subtensors on gzip files are not implemented."
            result = readNums(file_handle,
                              elem_type,
                              num_elems * elem_size).reshape(shape)
        elif subtensor is None:
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        elif isinstance(subtensor, slice):
            if subtensor.step not in (None, 1):
                raise NotImplementedError('slice with step',
                                          subtensor.step)
            if subtensor.start not in (None, 0):
                bytes_per_row = numpy.prod(shape[1:]) * elem_size
                file_handle.seek(
                    beginning + subtensor.start * bytes_per_row)
            shape[0] = min(shape[0], subtensor.stop) - subtensor.start
            num_elems = numpy.prod(shape)
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        else:
            raise NotImplementedError('subtensor access not written yet:',
                                      subtensor)

        return result
    fname = getPath(which_set)
    fname = datasetCache.cache_file(fname)
    file_handle = open(fname)

    return parseNORBFile(file_handle, subtensor)
    def get_topological_view(self, mat=None, single_tensor=True):
        """
        .. todo::
            WRITEME
        """
        result = super(SmallNORB, self).get_topological_view(mat)

        if single_tensor:
            warnings.warn("The single_tensor argument is True by default to "
                          "maintain backwards compatibility. This argument "
                          "will be removed, and the behavior will become that "
                          "of single_tensor=False, as of August 2014.")
            axes = list(self.view_converter.axes)
            s_index = axes.index('s')
            assert axes.index('b') == 0
            num_image_pairs = result[0].shape[0]
            shape = (num_image_pairs, ) + self.view_converter.shape

            # inserts a singleton dimension where the 's' dimesion will be
            mono_shape = shape[:s_index] + (1, ) + shape[(s_index + 1):]

            for i, res in enumerate(result):
                logger.info("result {0} shape: {1}".format(i, str(res.shape)))

            result = tuple(t.reshape(mono_shape) for t in result)
            result = numpy.concatenate(result, axis=s_index)
        else:
            warnings.warn("The single_tensor argument will be removed on "
                          "August 2014. The behavior will be the same as "
                          "single_tensor=False.")

        return result

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
def convertToCMYK(X, tile_shape) :
    R = X[0]
    G = X[1]
    B = X[2]
    
    R = tile_raster_images(X=R, img_shape=(32,32), tile_shape=tile_shape, tile_spacing=(1,1));
    G = tile_raster_images(X=G, img_shape=(32,32), tile_shape=tile_shape, tile_spacing=(1,1));
    B = tile_raster_images(X=B, img_shape=(32,32), tile_shape=tile_shape, tile_spacing=(1,1));

    RGBArray = numpy.zeros((len(R),len(R.T),3),'uint8');
    RGBArray[..., 0] = R
    RGBArray[..., 1] = G
    RGBArray[..., 2] = B
    return RGBArray

    """K = 1 - numpy.maximum(R,G,B)
    C = (1-R-K) / (1-K)
    M = (1-G-K) / (1-K)
    Y = (1-B-K) / (1-K)
    Final = numpy.zeros((4,len(K),len(K.T)))
    for c in xrange(4) :
        if c == 0 :
            source = K
        elif c == 1 :
            source = C
        elif c == 2 :
            source = M
        elif c == 3 :
            source = Y
        for x in xrange(len(K)) :
            for y in xrange(len(K.T)) :
                Final[c][x][y] = source[x][y]
    return tuple(Final)"""



def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        print len(X)
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                print X[i], 'recurrent'
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array