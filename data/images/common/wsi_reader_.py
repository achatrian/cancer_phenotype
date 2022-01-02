import numpy as np    
import cv2
import xml.etree.ElementTree as ET
import re
from fractions import Fraction

class WSIReader:
    def close(self):
        raise NotImplementedError
        
    @property
    def tile_dimensions(self):
        raise NotImplementedError
        
    def read_region(self, x_y, level, tile_size, normalize=True, downsample_level_0=False):
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        x, y = x_y
        if downsample_level_0 and level > 0:
            downsample = round(self.level_dimensions[0][0] / self.level_dimensions[level][0])
            x, y = x * downsample, y * downsample
            tile_w, tile_h = tile_size[0] * downsample, tile_size[1] * downsample
            width, height = self.level_dimensions[0] 
        else:
            tile_w, tile_h = tile_size
            width, height = self.level_dimensions[level]
            
        tile_w = tile_w + x if x < 0 else tile_w
        tile_h = tile_h + y if y < 0 else tile_h
        x = max(x, 0)
        y = max(y, 0)
        tile_w = width - x if (x + tile_w > width) else tile_w
        tile_h = height - y if (y + tile_h > height) else tile_h
        
        tile, alfa_mask = self._read_region((x,y), 0 if downsample_level_0 else level, (tile_w, tile_h))
        if downsample_level_0 and level > 0:
            tile_w = tile_w // downsample
            tile_h = tile_h // downsample
            x = x // downsample
            y = y // downsample
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_CUBIC).astype(np.bool)
        
        if normalize:
            tile = self._normalize(tile)
        
        padding = [(y-x_y[1],tile_size[1]-tile_h+min(x_y[1],0)), (x-x_y[0], tile_size[0]-tile_w+min(x_y[0],0))]
        tile = np.pad(tile, padding + [(0,0)]*(len(tile.shape)-2), 'constant', constant_values=0)
        alfa_mask = np.pad(alfa_mask, padding, 'constant', constant_values=0)
        
        return tile, alfa_mask
        
    def read_region_ds(self, x_y, downsample, tile_size, normalize=True, downsample_level_0=False):
        if not isinstance(downsample, int) or downsample <= 0:
            raise RuntimeError('Downsample factor must be a positive integer')
            
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
            
        if downsample == 1 or (downsample in self.level_downsamples and not downsample_level_0):
            level = self.get_best_level_for_downsample(downsample)
            tile, alfa_mask = self.read_region(x_y, level, tile_size, False, False)
        else:
            level = 0 if downsample_level_0 else self.get_best_level_for_downsample(downsample)
            x_y_level = [round(coord * downsample / self.level_downsamples[level]) for coord in x_y]
            tile_size_level = [round(dim * downsample / self.level_downsamples[level]) for dim in tile_size]
            tile, alfa_mask = self.read_region(x_y_level, level, tile_size_level, False, False)
            tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_CUBIC)
            alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), tile_size, interpolation=cv2.INTER_CUBIC).astype(np.bool)
        
        if normalize:
            tile = self._normalize(tile)
            
        return tile, alfa_mask

    def _read_region(self, x_y, level, tile_size):
        raise NotImplementedError
        
    def get_best_level_for_downsample(self, downsample):        
        if downsample < self.level_downsamples[0]:
            return 0

        for i in range(1, self.level_count):
            if downsample < self.level_downsamples[i]:
                return i - 1
        
        return self.level_count - 1
        
    def get_downsampled_slide(self, dims, normalize=True):
        downsample = min(a / b for a, b in zip(self.level_dimensions[0], dims))
        level = self.get_best_level_for_downsample(downsample)
        slide_downsampled, alfa_mask = self.read_region((0,0), level, self.level_dimensions[level], normalize=normalize)
        slide_downsampled = cv2.resize(slide_downsampled, dims, interpolation=cv2.INTER_CUBIC)
        alfa_mask = cv2.resize(alfa_mask.astype(np.uint8), dims, interpolation=cv2.INTER_CUBIC).astype(np.bool)
        return slide_downsampled, alfa_mask
        
    @property
    def level_dimensions(self):
        raise NotImplementedError
        
    @property
    def level_count(self):
        raise NotImplementedError  
        
    @property
    def mpp(self):
        raise NotImplementedError
        
    @property
    def dtype(self):
        raise NotImplementedError
        
    @property
    def n_channels(self):
        raise NotImplementedError
        
    @property
    def level_downsamples(self):
        if not hasattr(self, '_level_downsamples'):
            self._level_downsamples = []
            width, height = self.level_dimensions[0]
            for level in range(self.level_count):
                w, h = self.level_dimensions[level]
                ds = round(width / w)
                self._level_downsamples.append(ds)
        return self._level_downsamples
             
    @staticmethod 
    def _normalize(pixels):
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels
    
    @staticmethod
    def _round(x, base):
        return base * round(x/base)
        
class OpenSlideReader(WSIReader):
    def __init__(self, slide_path, **kwargs):
        import openslide
        self.slide_path = slide_path
        self._slide = openslide.open_slide(str(slide_path))
        
    def close(self):
        self.slide_path = None
        self._slide.close()
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
           
    def _read_region(self, x_y, level, tile_size):
        tile = np.array(self._slide.read_region(x_y, level, tile_size), dtype=np.uint8)
        alfa_mask = tile[:,:,3] > 0
        tile = tile[:,:,:3]
        return tile, alfa_mask
    
    def get_best_level_for_downsample(self, downsample):
        return self._slide.get_best_level_for_downsample(downsample)
        
    @property
    def level_dimensions(self):
        return self._slide.level_dimensions
        
    @property
    def level_count(self):
        return self._slide.level_count
        
    @property
    def mpp(self):
        return float(self._slide.properties['openslide.mpp-x']), float(self._slide.properties['openslide.mpp-y'])
        
    @property
    def dtype(self):
        return np.dtype(np.uint8)
       
    @property 
    def n_channels(self):
        return 3
        
    @property
    def level_downsamples(self):
        return self._slide.level_downsamples
        
    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = []
            for level in range(self.level_count):
                tile_width = self._slide.properties[f'openslide.level[{level}].tile-width']
                tile_height = self._slide.properties[f'openslide.level[{level}].tile-height']
                self._tile_dimensions.append((tilewidth, tilelength))
        return self._tile_dimensions
        
class TiffReader(WSIReader):
    def __init__(self, slide_path, series=0, **kwargs):
        import tifffile
        import zarr
        self.slide_path = slide_path
        self.series = series
        self._store = tifffile.imread(slide_path, aszarr=True, series=series)
        self._z = zarr.open(self._store, mode='r')

    def close(self):
        self.slide_path = None
        self._store.close()
        if hasattr(self, '_mpp'):
            delattr(self, '_mpp')
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
        if hasattr(self, '_level_dimensions'):
            delattr(self, '_level_dimensions')
        if hasattr(self, '_level_downsamples'):
            delattr(self, '_level_downsamples')
        
    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = []
            for level in range(self.level_count):
                page = self._store._data[level].pages[0]
                self._tile_dimensions.append((page.tilewidth, page.tilelength))
        return self._tile_dimensions
        
    def _read_region(self, x_y, level, tile_size):
        x, y = x_y
        tile_w, tile_h = tile_size
        return self._z[level][y:y+tile_h, x:x+tile_w], np.ones((tile_h, tile_w), np.bool)
        
    @property 
    def level_dimensions(self):
        if not hasattr(self, '_level_dimensions'):
            self._level_dimensions = []
            for level in range(self.level_count):
                page = self._store._data[level].pages[0]
                self._level_dimensions.append((page.imagewidth, page.imagelength))
        return self._level_dimensions
    
    @property    
    def level_count(self):
        return len(self._z)
        
    @property
    def mpp(self):
        if not hasattr(self, '_mpp'):
            self._mpp = None
            page = self._store._data[0].pages[0]
            if page.is_svs:
                metadata = tifffile.tifffile.svs_description_metadata(page.description)
                self._mpp = (metadata['MPP'], metadata['MPP'])
            elif page.is_ome:
                root = ET.fromstring(self._reader.ome_metadata)
                namespace = re.search('^{.*}', root.tag)
                namespace = namespace.group() if namespace else ''
                pixels = list(root.findall(namespace + 'Image'))[self.series].find(namespace + 'Pixels')
                self._mpp = (float(pixels.get('PhysicalSizeX')), float(pixels.get('PhysicalSizeY')))
            elif page.is_philips:
                root = ET.fromstring(self._reader.philips_metadata)
                mpp = float(root.find("./Attribute/[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject/[@ObjectType='DPScannedImage']/Attribute/[@Name='PIM_DP_IMAGE_TYPE'][.='WSI']/Attribute[@Name='PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE']/Array/DataObject[@ObjectType='PixelDataRepresentation']/Attribute[@Name='DICOM_PIXEL_SPACING']").text)
                self._mpp = (mpp, mpp)
            elif page.is_ndpi or page.is_scn or page.is_qpi or True:
                page = self._store._data[0].pages[0]
                if 'ResolutionUnit' in page.tags and page.tags['ResolutionUnit'].value == 3:
                    self._mpp = (1e4/float(Fraction(*page.tags['XResolution'].value)),\
                                 1e4/float(Fraction(*page.tags['YResolution'].value)))
        return self._mpp
    
    @property    
    def dtype(self):
        return self._z[0].dtype
    
    @property
    def n_channels(self):
        page = self._store._data[0].pages[0]
        return page.samplesperpixel
        
class IsyntaxReader(WSIReader):
    def __init__(self, slide_path, **kwargs):
        from pixelengine import PixelEngine
        from softwarerendercontext import SoftwareRenderContext
        from softwarerenderbackend import SoftwareRenderBackend
        self.slide_path = slide_path
        self._pe = PixelEngine(SoftwareRenderBackend(), SoftwareRenderContext())
        self._pe['in'].open(str(slide_path), 'ficom')
        self._view = self._pe['in']['WSI'].source_view
        trunc_bits = {0: [0, 0, 0]}
        self._view.truncation(False, False, trunc_bits)
        
    def close(self):
        self.slide_path = None
        self._pe['in'].close()
        if hasattr(self, '_tile_dimensions'):
            delattr(self, '_tile_dimensions')
        if hasattr(self, '_level_dimensions'):
            delattr(self, '_level_dimensions')
        if hasattr(self, '_level_downsamples'):
            delattr(self, '_level_downsamples')
        
    @property
    def tile_dimensions(self):
        if not hasattr(self, '_tile_dimensions'):
            self._tile_dimensions = [tuple(self._pe['in']['WSI'].block_size()[:2])] * self.level_count
        return self._tile_dimensions
        
    def _read_region(self, x_y, level, tile_size):
        x_start, y_start = x_y
        ds = self.level_downsamples[level]
        x_start *= ds
        y_start *= ds
        tile_w, tile_h = tile_size
        x_end, y_end = x_start + (tile_w-1)*ds, y_start + (tile_h-1)*ds
        view_range = [x_start, x_end, y_start, y_end, level]
        regions = self._view.request_regions([view_range],
                            self._view.data_envelopes(level),
                            True, [255,255,255], self._pe.BufferType(1))
        region, = self._pe.wait_any(regions)
        tile = np.empty(np.prod(tile_size)*4, dtype=np.uint8)
        region.get(tile)
        tile.shape = (tile_h, tile_w, 4)
        return tile[:,:,:3], tile[:,:,3] > 0

    @property
    def level_dimensions(self):
        if not hasattr(self, '_level_dimensions'):
            self._level_dimensions = []
            for level in range(self.level_count):
                x_step, x_end = self._view.dimension_ranges(level)[0][1:]
                y_step, y_end = self._view.dimension_ranges(level)[1][1:]
                range_x = (x_end + 1) // x_step
                range_y = (y_end + 1) // y_step
                self._level_dimensions.append((range_x, range_y))
        return self._level_dimensions
        
    @property
    def level_count(self):
        return self._view.num_derived_levels + 1
        
    @property
    def mpp(self):
        return self._view.scale[0], self._view.scale[1]
        
    @property
    def dtype(self):
        return np.dtype(np.uint8)
        
    @property
    def n_channels(self):
        return 3
        
    @property
    def level_downsamples(self):
        if not hasattr(self, '_level_downsamples'):
            self._level_downsamples = [self._view.dimension_ranges(level)[0][1] for level in range(self.level_count)]
        return self._level_downsamples


def get_reader_impl(slide_path):
    if slide_path.suffix == '.isyntax':
        return IsyntaxReader
    else:
        return TiffReader
            
