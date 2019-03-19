import os
import glob
from optparse import OptionParser
from deepzoom_tile import DeepZoomStaticTiler

r"Creates deepzoom images for all the WSIs in a folder"


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <folder>')
    parser.add_option('-B', '--ignore-bounds', dest='limit_bounds',
                default=True, action='store_false',
                help='display entire scan area')
    parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
                type='int', default=1,
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
                default='jpeg',
                help='image format for tiles [jpeg]')
    parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
                type='int', default=4,
                help='number of worker processes to start [4]')
    parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
                type='int', default=90,
                help='JPEG compression quality [90]')
    parser.add_option('-r', '--viewer', dest='with_viewer',
                action='store_true',
                help='generate directory tree with HTML viewer')
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
                type='int', default=254,
                help='tile size [254]')

    (opts, args) = parser.parse_args()
    try:
        slide_dir_path = args[0]
    except IndexError:
        parser.error('Missing data folder argument')

    slide_paths = list(glob.glob(os.path.join(slide_dir_path, '**/*.svs'), recursive=True))
    slide_paths += list(glob.glob(os.path.join(slide_dir_path, '**/*.ndpi'), recursive=True))

    for slide_path in slide_paths:
        print("Making deepzoom for " + os.path.basename(slide_path))
        opts.basename = os.path.splitext(os.path.basename(slide_path))[0]
        DeepZoomStaticTiler(slide_path, opts.basename, opts.format,
                    opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality,
                    opts.workers, opts.with_viewer).run()
