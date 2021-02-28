try:
    from functools import cached_property
    property = cached_property
except ImportError:
    pass
import openslide
from openslide import OpenSlideError
import numpy as np
import tifffile
import cv2
from base.utils import debug


class BaseWSIReader:
    base_level_coordinates = False
    # determines whether region is read at coordinates relative to base level even when reading from higher levels

    @property
    def properties(self):
        r"""Get MPP and other slide metadata if available"""
        return {}

    def close(self):
        pass

    # def __del__(self):  # FIXME problems with calling close for tifffreader -- WSIReader loses attribute 'reader' at deletion
    #     self.close()

    def get_tile_dims(self, level):
        pass
        
    def read_region(self, x_y, level, size, read_base_level=False):
        pass

    def get_downsampled_slide(self, downsampling):
        pass

    def get_thumbnail(self, size):
        pass
        
    def get_dimensions(self, level):
        return ()
        
    def get_level_count(self):
        return 0
        
    def get_dtype(self):
        pass
        
    def get_n_channels(self):
        return 0

    @property
    def level_dimensions(self):
        return tuple(self.get_dimensions(level) for level in range(self.get_level_count()))

    @property
    def level_downsamples(self):
        return tuple(round(self.get_dimensions(0)[0] / self.get_dimensions(level)[0])
                     for level in range(self.get_level_count()))


class TiffBaseReader(BaseWSIReader):
    base_level_coordinates = True

    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.reader = tifffile.TiffFile(str(slide_path))
        if not self.reader.series:
            raise IOError("Corrupted file")

    def close(self):
        self.slide_path = None
        self.reader.close()
        self.reader = None

    def get_tile_dims(self, level):
        page = self.reader.series[0].levels[level].pages[0]
        return page.tilewidth, page.tilelength

    def read_region(self, x_y, level, size, read_base_level=False):
        x, y = x_y
        downsampling = round(self.get_dimensions(0)[0] / self.get_dimensions(level)[0])
        if self.base_level_coordinates:
            x *= downsampling
            y *= downsampling

        if level > 0 and read_base_level:
            downsampling = round(self.get_dimensions(0)[0] / self.get_dimensions(level)[0])
            x, y = x * downsampling, y * downsampling
            tile_w, tile_h = size[0] * downsampling, size[1] * downsampling
            width, height = self.get_dimensions(0)
            tile_w = width - x if (x + tile_w > width) else tile_w
            tile_h = height - y if (y + tile_h > height) else tile_h
            tile = self._read_region((x, y), 0, (tile_w, tile_h))
            tile_w = tile_w // downsampling
            tile_h = tile_h // downsampling
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
        else:
            tile_w, tile_h = size
            width, height = self.get_dimensions(level)
            tile_w = width - x if (x + tile_w > width) else tile_w
            tile_h = height - y if (y + tile_h > height) else tile_h
            tile = self._read_region((x, y), level, (tile_w, tile_h))

        if len(tile.shape) == 3 and tile.shape[2] == 4:
            alfa_mask = tile[:, :, 3] == 1
            tile = tile[:, :, :3]
        else:
            alfa_mask = np.ones(tile.shape[:2], dtype=int)*255

        tile = np.pad(tile, ((0, size[1] - tile_h), (0, size[0] - tile_w), (0, 0)), 'constant',
                      constant_values=0)
        alfa_mask = np.pad(alfa_mask, ((0, size[1] - tile_h), (0, size[0] - tile_w)), 'constant',
                           constant_values=0)
        tile = np.concatenate((tile, alfa_mask[..., np.newaxis]), axis=2)
        return tile

    def _read_region(self, x_y, level, tile_size):
        page = self.reader.series[0].levels[level].pages[0]

        if not page.is_tiled:
            raise ValueError("Input page must be tiled.")

        x, y = x_y
        j0, i0 = x_y
        w, h = tile_size

        im_width = page.imagewidth
        im_height = page.imagelength

        # w = im_width - x if (x + w > im_width) else w
        # h = im_height - y if (y + h > im_height) else h

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        if i0 < 0 or j0 < 0 or i0 + h > im_height or j0 + w > im_width:
            print((i0, j0, im_height, im_width))
            raise ValueError("Requested crop area is out of image bounds.")

        tile_width, tile_height = page.tilewidth, page.tilelength
        i1, j1 = i0 + h, j0 + w

        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.zeros((page.imagedepth,
                        (tile_i1 - tile_i0) * tile_height,
                        (tile_j1 - tile_j0) * tile_width,
                        page.samplesperpixel), dtype=page.dtype)

        fh = page.parent.filehandle

        jpegtables = page.tags.get('JPEGTables', None)
        if jpegtables is not None:
            jpegtables = jpegtables.value

        for i in range(tile_i0, tile_i1):
            for j in range(tile_j0, tile_j1):
                index = int(i * tile_per_line + j)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                if bytecount == 0:
                    continue

                fh.seek(offset)
                data = fh.read(bytecount)

                tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)
                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        tile = out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]
        tile = tile[0, :, :, :]

        return tile

    def _get_best_level_for_downsample(self, downsampling):
        width, height = self.get_dimensions(0)
        levels_downsample = []
        for level in range(self.get_level_count()):
            w, h = self.get_dimensions(level)
            ds = round(width / w)
            levels_downsample.append(ds)

        if downsampling < levels_downsample[0]:
            return 0

        for i in range(1, self.get_level_count()):
            if downsampling < levels_downsample[i]:
                return i - 1

        return self.get_level_count() - 1

    def get_downsampled_slide(self, downsampling):
        level = self._get_best_level_for_downsample(downsampling)
        page = self.reader.series[0].levels[level].pages[0]
        if any(dim > 15000 for dim in self.get_dimensions(level)):
            raise ValueError(f"Dimensions of thumbnail level are too large ({self.get_dimensions(level)})")
        if page.is_tiled:
            slide_downsampled = self.read_region((0, 0), level, self.get_dimensions(level))
        else:
            slide_downsampled = page.asarray()
            if len(slide_downsampled.shape) == 3 and slide_downsampled.shape[2] == 4:
                alfa_mask = slide_downsampled[:, :, 3] == 1
                slide_downsampled = slide_downsampled[:, :, :3]
            else:
                alfa_mask = np.ones(slide_downsampled.shape[:2], dtype=float)
            slide_downsampled = np.concatenate((slide_downsampled, alfa_mask[..., np.newaxis]), axis=2)
        return slide_downsampled

    def get_thumbnail(self, size):
        w, h = size
        w_base, h_base = self.get_dimensions(0)
        if w_base/w != h_base/h:
            raise ValueError("Input size must keep aspect ratio")
        downsampling = w_base/w
        return self.get_downsampled_slide(downsampling)

    def get_dimensions(self, level):
        if hasattr(self.reader.series[0], 'levels'):
            page = self.reader.series[0].levels[level].pages[0]
        else:
            if level != 0:
                raise ValueError("tiff file is not pyramidal (level must be 0)")
            page = self.reader.series[0].pages[0]
        return page.imagewidth, page.imagelength

    def get_level_count(self):
        if hasattr(self.reader.series[0], 'levels'):
            return len(self.reader.series[0].levels)
        else:
            return 1

    def get_dtype(self):
        return self.reader.series[0].levels[0].pages[0].dtype

    def get_n_channels(self):
        channels = self.reader.series[0].levels[0].pages[0].samplesperpixel
        return channels - 1 if channels == 4 else channels


class OpenSlideReaderBase(BaseWSIReader):
    base_level_coordinates = True

    def __init__(self,  slide_path):
        self.slide_path = slide_path
        self.slide = openslide.open_slide(str(slide_path))

    def close(self):
        self.slide_path = None
        self.slide.close()
        self.slide = None
    
    def get_tile_dims(self, level):
        tile_width = self.slide.properties[f'openslide.level[{level}].tile-width']
        tile_height = self.slide.properties[f'openslide.level[{level}].tile-height']
        return tile_width, tile_height
        
    def read_region(self, x_y, level, size, read_base_level=False):
        downsampling = round(self.slide.level_downsamples[level])
        x, y = x_y
        x_y = (x * downsampling, y * downsampling)
        tile = None
        alfa_mask = None
        try:
            if level > 0 and read_base_level:
                tile = np.array(
                    self.slide.read_region(x_y, 0, (size[0] * downsampling, size[1] * downsampling), ), dtype=np.uint8)
                tile = cv2.resize(tile, size, interpolation=cv2.INTER_CUBIC)
            else:
                tile = np.array(self.slide.read_region(x_y, level, size, ), dtype=np.uint8)
            alfa_mask = tile[:,:,3] == 1
            tile = tile[:,:,:3]
        except OpenSlideError:
            #handle tiles without data
            self.slide = openslide.open_slide(str(self.slide_path))
        finally:
            return tile, alfa_mask

    def get_downsampled_slide(self, downsampling):
        slide_downsampled = None
        while slide_downsampled is None:
            level = self.slide.get_best_level_for_downsample(downsampling)
            dimensions = self.slide.level_dimensions[level]
            try:
                slide_downsampled = np.array(self.slide.read_region((0, 0), level, dimensions, ))
            except OpenSlideError:
                self.slide = openslide.open_slide(str(self.slide_path))
                downsampling = downsampling*2
        return slide_downsampled[:,:,:3], slide_downsampled[:,:,3] == 1
    
    def get_dimensions(self, level):
        return self.slide.level_dimensions[level]
        
    def get_level_count(self):
        return self.slide.level_count
        
    def get_dtype(self):
        return np.dtype(np.uint8)
        
    def get_n_channels(self):
        return 3


class BioFormatsReaderBase(BaseWSIReader):
    base_level_coordinates = False

    def __init__(self, slide_path):
        import bioformats
        import javabridge as jutil
        jutil.start_vm(class_path=bioformats.JARS, run_headless=True)
        self.FormatTools = bioformats.formatreader.make_format_tools_class()
        self.slide_path = slide_path
        self.reader = bioformats.ImageReader(str(self.slide_path), perform_init=False)
        self.reader.rdr.setFlattenedResolutions(True)
        self.reader.init_reader()
        
    def close(self):
        self.slide_path = None
        self.reader.close()
        self.reader = None
        
    def get_tile_dims(self, level):
        rdr = self.reader.rdr
        rdr.setSeries(level)
        return rdr.getOptimalTileWidth(), rdr.getOptimalTileheight()
        
    def read_region(self, x_y, level, size, read_base_level=False):
        x, y = x_y
        if level > 0 and read_base_level:
            downsampling = round(self.get_dimensions(0)[0] / self.get_dimensions(level)[0])
            x, y = x * downsampling, y * downsampling
            tile_w, tile_h = size[0] * downsampling, size[1] * downsampling
            width, height = self.get_dimensions(0)
            tile_w = width - x if (x + tile_w > width) else tile_w
            tile_h = height - y if (y + tile_h > height) else tile_h
            tile = self.reader.read(series=level, XYWH=(x, y, tile_w, tile_h), rescale=True)
            tile_w = tile_w // downsampling
            tile_h = tile_h // downsampling
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
        else:       
            tile_w, tile_h = size
            width, height = self.get_dimensions(level)
            tile_w = width - x if (x + tile_w > width) else tile_w
            tile_h = height - y if (y + tile_h > height) else tile_h
            tile = self.reader.read(series=level, XYWH=(x, y, tile_w, tile_h), rescale=True)
        
        if len(tile.shape) == 3 and tile.shape[2] == 4:
            alfa_mask = tile[:,:,3] == 1
            tile = tile[:,:,:3]
        else:
            alfa_mask = np.ones(tile.shape[:2], dtype=bool)
        
        tile = np.pad(tile, ((0, size[1] - tile_h), (0, size[0] - tile_w), (0, 0)), 'constant', constant_values=0)
        alfa_mask = np.pad(alfa_mask, ((0, size[1] - tile_h), (0, size[0] - tile_w)), 'constant', constant_values=0)
        
        return tile, alfa_mask
        
    def _get_best_level_for_downsample(self, downsampling):
        width, height = self.get_dimensions(0)
        levels_downsample = []
        for level in range(self.get_level_count()):
            w, h = self.get_dimensions(level)
            ds = round(width / w)
            levels_downsample.append(ds) 
            
        if downsampling < levels_downsample[0]:
            return 0

        for i in range(1, self.get_level_count()):
            if downsampling < levels_downsample[i]:
                return i - 1
        
        return self.get_level_count() - 1

    def get_downsampled_slide(self, downsampling):
        level = self._get_best_level_for_downsample(downsampling)
        slide_downsampled = bioformats.load_image(str(self.slide_path), series=level, rescale=True)
        if len(slide_downsampled.shape) == 3 and slide_downsampled.shape[2] == 4:
            alfa_mask = slide_downsampled[:,:,3] == 1
            slide_downsampled = slide_downsampled[:,:,:3]
        else:
            alfa_mask = np.ones(slide_downsampled.shape[:2], dtype=bool)
        return slide_downsampled, alfa_mask
        
    def get_dimensions(self, level):
        self.reader.rdr.setSeries(level)
        return self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY()
        
    def get_level_count(self):
        return self.reader.rdr.getSeriesCount()
        
    def get_dtype(self):
        pixel_type = self.reader.rdr.getPixelType()
        little_endian = self.reader.rdr.isLittleEndian()
        if pixel_type == self.FormatTools.INT8:
            dtype = np.int8
        elif pixel_type == self.FormatTools.UINT8:
            dtype = np.uint8
        elif pixel_type == self.FormatTools.UINT16:
            dtype = '<u2' if little_endian else '>u2'
        elif pixel_type == self.FormatTools.INT16:
            dtype = '<i2' if little_endian else '>i2'
        elif pixel_type == self.FormatTools.UINT32:
            dtype = '<u4' if little_endian else '>u4'
        elif pixel_type == self.FormatTools.INT32:
            dtype = '<i4' if little_endian else '>i4'
        elif pixel_type == self.FormatTools.FLOAT:
            dtype = '<f4' if little_endian else '>f4'
        elif pixel_type == self.FormatTools.DOUBLE:
            dtype = '<f8' if little_endian else '>f8'
        return np.dtype(dtype)
        
    def get_n_channels(self):
        channels = self.reader.rdr.getSizeC()
        return channels - 1 if channels == 4 else channels


def get_reader(slide_path):
    if slide_path.suffix == '.tiff' or slide_path.suffix == '.tif' or slide_path.suffix == '.svs':
        return TiffBaseReader
    return BioFormatsReaderBase


# def make_iformat_reader_class():
#
#     '''Bind a Java class that implements IFormatReader to a Python class
#     Returns a class that implements IFormatReader through calls to the
#     implemented class passed in. The returned class can be subclassed to
#     provide additional bindings.
#     '''
#     import bioformats
#     import javabridge as jutil
#     jutil.start_vm(class_path=bioformats.JARS, run_headless=True)
#     class IFormatReader(object):
#         '''A wrapper for loci.formats.IFormatReader
#         See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
#         '''
#         close = jutil.make_method('close','()V',
#                                   'Close the currently open file and free memory')
#         getDimensionOrder = jutil.make_method('getDimensionOrder',
#                                               '()Ljava/lang/String;',
#                                               'Return the dimension order as a five-character string, e.g. "XYCZT"')
#         getGlobalMetadata = jutil.make_method('getGlobalMetadata',
#                                         '()Ljava/util/Hashtable;',
#                                         'Obtains the hashtable containing the global metadata field/value pairs')
#         getMetadata = getGlobalMetadata
#         getMetadataValue = jutil.make_method('getMetadataValue',
#                                              '(Ljava/lang/String;)'
#                                              'Ljava/lang/Object;',
#                                              'Look up a specific metadata value from the store')
#         getSeriesMetadata = jutil.make_method('getSeriesMetadata',
#                                               '()Ljava/util/Hashtable;',
#                                               'Obtains the hashtable contaning the series metadata field/value pairs')
#         getSeriesCount = jutil.make_method('getSeriesCount',
#                                            '()I',
#                                            'Return the # of image series in the file')
#         getSeries = jutil.make_method('getSeries', '()I',
#                                       'Return the currently selected image series')
#         getImageCount = jutil.make_method('getImageCount',
#                                           '()I','Determines the number of images in the current file')
#         getIndex = jutil.make_method('getIndex', '(III)I',
#                                      'Get the plane index given z, c, t')
#         getOptimalTileheight = jutil.make_method('getOptimalTileheight', '()I',
#                                      'Get the optimal sub-image height for use with openBytes')
#         getOptimalTileWidth = jutil.make_method('getOptimalTileWidth', '()I',
#                                      'Get the optimal sub-image width for use with openBytes')
#         getRGBChannelCount = jutil.make_method('getRGBChannelCount',
#                                                '()I','Gets the number of channels per RGB image (if not RGB, this returns 1')
#         getSizeC = jutil.make_method('getSizeC', '()I',
#                                      'Get the number of color planes')
#         getSizeT = jutil.make_method('getSizeT', '()I',
#                                      'Get the number of frames in the image')
#         getSizeX = jutil.make_method('getSizeX', '()I',
#                                      'Get the image width')
#         getSizeY = jutil.make_method('getSizeY', '()I',
#                                      'Get the image height')
#         getSizeZ = jutil.make_method('getSizeZ', '()I',
#                                      'Get the image depth')
#         getPixelType = jutil.make_method('getPixelType', '()I',
#                                          'Get the pixel type: see FormatTools for types')
#         isLittleEndian = jutil.make_method('isLittleEndian',
#                                            '()Z','Return True if the data is in little endian order')
#         isRGB = jutil.make_method('isRGB', '()Z',
#                                   'Return True if images in the file are RGB')
#         isInterleaved = jutil.make_method('isInterleaved', '()Z',
#                                           'Return True if image colors are interleaved within a plane')
#         isIndexed = jutil.make_method('isIndexed', '()Z',
#                                       'Return True if the raw data is indexes in a lookup table')
#         openBytes = jutil.make_method('openBytes','(I)[B',
#                                       'Get the specified image plane as a byte array')
#         openBytesXYWH = jutil.make_method('openBytes','(IIIII)[B',
#                                           '''Get the specified image plane as a byte array
#                                           (corresponds to openBytes(int no, int x, int y, int w, int h))
#                                           no - image plane number
#                                           x,y - offset into image
#                                           w,h - dimensions of image to return''')
#         setSeries = jutil.make_method('setSeries','(I)V','Set the currently selected image series')
#         setGroupFiles = jutil.make_method('setGroupFiles', '(Z)V',
#                                           'Force reader to group or not to group files in a multi-file set')
#         setMetadataStore = jutil.make_method('setMetadataStore',
#                                              '(Lloci/formats/meta/MetadataStore;)V',
#                                              'Sets the default metadata store for this reader.')
#         setMetadataOptions = jutil.make_method('setMetadataOptions',
#                                                '(Lloci/formats/in/MetadataOptions;)V',
#                                                'Sets the metadata options used when reading metadata')
#         setResolution = jutil.make_method(
#             'setResolution',
#             '()I',
#             'Set the resolution level'
#         )
#         getResolutionCount = jutil.make_method(
#             'getResolutionCount',
#             '()I',
#             'Return the number of resolutions for the current series'
#         )
#         setFlattenedResolutions = jutil.make_method(
#             'setFlattenedResolutions',
#             '(Z)V',
#             'Set whether or not to flatten resolutions into individual series'
#         )
#         getDimensionOrder = jutil.make_method(
#             'getDimensionOrder',
#             '()Ljava/lang/String;',
#             'Return the dimension order as a five-character string, e.g. "XYCZT"'
#         )
#         isThisTypeS = jutil.make_method(
#             'isThisType',
#             '(Ljava/lang/String;)Z',
#             'Return true if the filename might be handled by this reader')
#         isThisTypeSZ = jutil.make_method(
#             'isThisType',
#             '(Ljava/lang/String;Z)Z',
#             '''Return true if the named file is handled by this reader.
#             filename - name of file
#             allowOpen - True if the reader is allowed to open files
#                         when making its determination
#             ''')
#         isThisTypeStream = jutil.make_method(
#             'isThisType',
#             '(Lloci/common/RandomAccessInputStream;)Z',
#             '''Return true if the stream might be parseable by this reader.
#             stream - the RandomAccessInputStream to be used to read the file contents
#             Note that both isThisTypeS and isThisTypeStream must return true
#             for the type to truly be handled.''')
#         def setId(self, path):
#             '''Set the name of the file'''
#             jutil.call(self.o, 'setId',
#                        '(Ljava/lang/String;)V',
#                        path)
#
#         getMetadataStore = jutil.make_method('getMetadataStore', '()Lloci/formats/meta/MetadataStore;',
#                                              'Retrieves the current metadata store for this reader.')
#         get8BitLookupTable = jutil.make_method(
#             'get8BitLookupTable',
#             '()[[B', 'Get a lookup table for 8-bit indexed images')
#         get16BitLookupTable = jutil.make_method(
#             'get16BitLookupTable',
#             '()[[S', 'Get a lookup table for 16-bit indexed images')
#         def get_class_name(self):
#             return jutil.call(jutil.call(self.o, 'getClass', '()Ljava/lang/Class;'),
#                               'getName', '()Ljava/lang/String;')
#
#         @property
#         def suffixNecessary(self):
#             if self.get_class_name() == 'loci.formats.in.JPKReader':
#                 return True;
#             env = jutil.get_env()
#             klass = env.get_object_class(self.o)
#             field_id = env.get_field_id(klass, "suffixNecessary", "Z")
#             if field_id is None:
#                 return None
#             return env.get_boolean_field(self.o, field_id)
#
#         @property
#         def suffixSufficient(self):
#             if self.get_class_name() == 'loci.formats.in.JPKReader':
#                 return True;
#             env = jutil.get_env()
#             klass = env.get_object_class(self.o)
#             field_id = env.get_field_id(klass, "suffixSufficient", "Z")
#             if field_id is None:
#                 return None
#             return env.get_boolean_field(self.o, field_id)
#
#     return IFormatReader
#
# bioformats.formatreader.make_iformat_reader_class = make_iformat_reader_class
#
