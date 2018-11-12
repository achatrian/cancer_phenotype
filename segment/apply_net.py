
#Apply network to whole WSI and save:
import os
import argparse
import time
import timeit
from itertools import product

import numpy as np
from torch import load, stack, cuda, no_grad
from torch.autograd import Variable
import openslide
import asyncio
import imageio
from cv2 import cvtColor, COLOR_RGB2GRAY

from utils import get_flags, colorize

from models import *

#"An openslide_t object can be used concurrently from multiple threads
#without locking. (But you must lock or otherwise use memory barriers
#when passing the object between threads.)"  (but this is for C
# if io_bound:
#     if io_very_slow:
#         print("Use Asyncio")
#     else:
#        print("Use Threads")
# else:
#     print("Multi Processing")

def check_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass


"""
class SegMapMaker():

    def __init__(self, net, slidepath, savedir, level, regionshape):
        self.net = net.cuda() if cuda.is_available() else net
        self.net.eval()
        self.slide = openslide.open_slide(slidepath)
        self.slidename = slidepath.split('/')[-1]
        self.savedir = savedir
        self.slideshape = self.slide.level_dimensions[level]
        self.level = level
        self.regionshape = regionshape
        self.xyin = [(x,y) for y in range(0, self.slideshape[0] - (self.slideshape[0] % regionshape[0]), regionshape[0]) \
                                for x in range(0, self.slideshape[1] - (self.slideshape[1] % regionshape[0]), regionshape[0])]
        self.regions = [] #list for temporarily storing WSI regions
        self.seg_maps = [] #list for temporarily storing segmentation maps
        self.xyout = [] #output coordinates
        def timefunc(): region = self.make_region((0,0))
        self.batch_size = 10
        self.readtime = timeit.Timer(timefunc).timeit(number=self.batch_size)


    def get_tasks(self):
        tasks = [self.get_regions(), self.feed_regions(), self.save_seg_maps()]
        return tasks

    def to_tensor(self, img):
        img = img[np.newaxis, :, :]
        tensor = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        return tensor

    def make_region(self, xy):
        region = self.slide.read_region(xy, self.level, self.regionshape).convert('RGB') #takes a few seconds
        region = cvtColor(np.array(region), COLOR_RGB2GRAY)
        return self.to_tensor(region).cuda if cuda.is_available() else self.to_tensor(region)

    async def get_regions(self):
        self.clock = time.time()
        iread = 0
        while self.xyin: #run until all coordinates have been processed
            xy = self.xyin.pop()
            region = self.make_region(xy)
            self.regions.append((region, xy))
            iread += 1
            if iread % self.batch_size == 0:
                print("Read {} images".format(iread))
                await asyncio.wait(1)

    async def call_net(self,input):
        print("Feeding {} images".format(input.shape[0]))
        return self.net(input)


    async def feed_regions(self):
        await asyncio.sleep(self.readtime) #so that at least one region is read in
        self.clock = time.time() - self.clock
        while self.regions or self.xyin:
            if self.regions and len(self.xyout) < self.batch_size:
                nowlen = len(self.regions)
                tofeed, xyfeed = [region[0] for region in self.regions[:nowlen]], \
                                        [region[1] for region in self.regions[:nowlen]]
                del self.regions[:nowlen]
                with no_grad():
                    input = Variable(stack(tofeed))
                    seg_map = await self.call_net(input)
                seg_map_np = seg_map.cpu().numpy()
                self.seg_maps.extend(seg_map_np) #iterates over first axis (batch), so get images
                self.xyout.extend(xyfeed)
            else:
                await asyncio.sleep(self.readtime)
                self.clock = time.time() - self.clock
                print("Not ready at {:.2f}".format(self.clock))

    async def save_seg_maps(self):
        await asyncio.sleep(self.readtime)
        self.clock = time.time() - self.clock
        isaved = 0
        while self.seg_maps or self.regions or self.xyout:
            if self.seg_maps:
                img, xy = self.seg_maps.pop(), self.xyout.pop()
                img = img.transpose(1,2,0)
                imgfile = os.path.join(self.savedir, self.slidename)
                imageio.imwrite(imgfile, img)
                isaved += 1
                if isaved % 1 == 0:
                    print("Saved {} images".format(isaved))
                    await asyncio.wait(10)
            else:
                await asyncio.sleep(self.readtime)
                self.clock = time.time() - self.clock
                print("Not ready at {:.2f}".format(self.clock))
"""

class SegMapMaker():

    def __init__(self, net, slidepath, savedir, level, regionshape):
        self.net = net.cuda() if cuda.is_available() else net
        self.net.eval()
        self.slide = openslide.open_slide(slidepath)
        self.slidename = slidepath.split('/')[-1]
        self.savedir = savedir
        self.slideshape = self.slide.level_dimensions[level]
        self.level = level
        self.regionshape = regionshape
        self.xyin = [(x,y) for y in range(0, self.slideshape[0] - (self.slideshape[0] % regionshape[0]), regionshape[0]) \
                                for x in range(0, self.slideshape[1] - (self.slideshape[1] % regionshape[0]), regionshape[0])]
        self.regions = [] #list for temporarily storing WSI regions
        self.seg_maps = [] #list for temporarily storing segmentation maps
        self.xyout = [] #output coordinates
        self.batch_size = 10
        #def timefunc(): region = self.make_region((0,0))
        #self.readtime = timeit.Timer(timefunc).timeit(number=self.batch_size)

    def to_tensor(self, img):
        img = img[np.newaxis, :, :]
        tensor = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        return tensor

    def make_region(self, xy):
        region = self.slide.read_region(xy, self.level, self.regionshape).convert('RGB') #takes a few seconds
        region = cvtColor(np.array(region), COLOR_RGB2GRAY)
        return self.to_tensor(region).cuda if cuda.is_available() else self.to_tensor(region)

    def start(self):
        self.clock = time.time()
        iread = 0
        while self.xyin: #run until all coordinates have been processed
            xy = self.xyin.pop()
            region = self.make_region(xy)
            self.regions.append((region, xy))
            iread += 1
            if iread % self.batch_size == 0:
                self.clock = time.time()
                self.feed_regions()
                self.save_seg_maps()
                self.update_clock()
                print("Saved {} images in {:.1g}s".format(self.batch_size, self.clock))

    def feed_regions(self):
        while self.regions:
            nowlen = len(self.regions)
            tofeed, xyfeed = [region[0] for region in self.regions[:nowlen]], \
                                    [region[1] for region in self.regions[:nowlen]]
            del self.regions[:nowlen]
            with no_grad():
                input = Variable(stack(tofeed))
                seg_map = self.net(input)
            seg_map_np = seg_map.cpu().numpy()
            self.seg_maps.extend(seg_map_np) #iterates over first axis (batch), so get images
            self.xyout.extend(xyfeed)

    def save_seg_maps(self):
        while self.seg_maps:
            pred, xy = self.seg_maps.pop(), self.xyout.pop()
            img = colorize(pred)
            imgfile = os.path.join(self.savedir, self.slidename) + "{},{}.png".format(*xy) #error if not png
            imageio.imwrite(imgfile, img)

    def update_clock(self): self.clock = time.time() - self.clock

    def merge_regions(self, mergenums=(10,10)):
        bigshape = tuple(min(dim * n, dim_s) for dim, n, dim_s in zip(self.regionshape, mergenums, self.slideshape))
        bigimg = np.zeros(bigshape)
        self.clock = time.time()
        slide_prefix = os.path.join(self.savedir, self.slidename)
        for i, j in product(range(mergenums[0]), range(mergenums[1])):
            region = imageio.imread(slide_prefix + "{},{}.png".format(*xy))
            bigimg[big]


#TODO take into account regions on ends of slide: (resample pixels till regions are 1024x1024 again

def main(FLAGS):

    TRAINFLAGS = get_flags(FLAGS.flags_filepath)  #get training arguments for loading net etc.
    inputs = {'num_classes' : TRAINFLAGS.num_class, 'num_channels' : TRAINFLAGS.num_filters,
                'grayscale' : TRAINFLAGS.grayscale, 'batchnorm' : False}
    #Setup
    if TRAINFLAGS.network_id == "UNet1":
        net = UNet1(**inputs).cuda() if cuda.is_available() else UNet1(**inputs) #possible classes are stroma, gland, lumen
    elif TRAINFLAGS.network_id == "UNet2":
        net = UNet2(**inputs).cuda() if cuda.is_available() else UNet2(**inputs)
    netdict = load(FLAGS.model_filepath, map_location = None if cuda.is_available() else'cpu')
    netdict = {entry[7:] : tensor for entry, tensor in netdict.items()} #NB there is a .module at beginning that is quite annyoing and needs to be removed ...
    net.load_state_dict(netdict)
    level = int(TRAINFLAGS.downsample / 2.0)
    regionshape = (TRAINFLAGS.image_size,) * 2
    savedir = '/'.join(FLAGS.model_filepath.split('/')[:-1] + [FLAGS.slidepath.split('/')[-1]])
    check_mkdir(savedir)
    segmapmaker = SegMapMaker(net, FLAGS.slidepath, savedir, level, regionshape)

    #Run slide through network and save results
    start = time.time()
    segmapmaker.start()
    end = time.time()
    print("Total time: {}".format(end - start))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', "--model_filepath", required = True, type=str)
    parser.add_argument('-ff', "--flags_filepath", default=None, type=str)
    parser.add_argument('-s', "--slidepath", required = True, type=str)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
