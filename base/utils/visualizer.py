import os
import sys
import ntpath
import time
from collections import OrderedDict
import numpy as np
from scipy.misc import imresize
from . import utils
from . import html

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = utils.tensor2im(im_data, label.endswith("_map"))
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        utils.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.experiment_name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, self.opt.experiment_name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, self.opt.experiment_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        raise ConnectionError("Could not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n")

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = utils.tensor2im(image[0, ...], label.endswith("_map"))
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                assert all([images[0].shape == image.shape for image in images]); "Ensure all images have same shape"
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 2,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 3,
                                  opts=dict(title=title + ' labels'))  # uses text to input table html
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = utils.tensor2im(image, label.endswith("_map"))
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx + 1)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = utils.tensor2im(image[0, ...], label.endswith("_map"))
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                utils.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image in visuals.items():
                    image_numpy = utils.tensor2im(image[0, ...], label.endswith("_map"))
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses_metrics(self, epoch, epoch_progress, losses, metrics):
        """
        plot both metrics and losses
        :param epoch:
        :param epoch_progress:
        :param losses:
        :param metrics:
        :return:
        """
        if not hasattr(self, 'loss_data'):
            losses_legend = list(losses.keys()) + [loss + "_val" for loss in losses.keys() if not loss.endswith('_val')]
            self.loss_data = {'X': [epoch - 1], 'Y': [[0] * len(losses_legend)], 'legend': losses_legend}
        self.loss_data['X'].append(epoch + epoch_progress)
        # fill with latest value if loss is not given for update
        self.loss_data['Y'].append([losses[k] if k in losses.keys() else self.loss_data['Y'][-1][i]
                       for i, k in enumerate(self.loss_data['legend'])])
        if not hasattr(self, 'metric_data'):
            metrics_legend = list(metrics.keys()) + [metric + "_val" for metric in metrics.keys()]
            self.metric_data = {'X': [epoch - 1], 'Y': [[0] * len(metrics_legend)], 'legend': metrics_legend}
        self.metric_data['X'].append(epoch + epoch_progress)
        self.metric_data['Y'].append([metrics[k] if k in metrics.keys() else self.metric_data['Y'][-1][j]
                                      for j, k in enumerate(self.metric_data['legend'])])
        try:
            self.vis.line(
                X=np.stack([np.array(self.loss_data['X'])] * len(self.loss_data['legend']), 1),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
            self.vis.line(
                X=np.stack([np.array(self.metric_data['X'])] * len(self.metric_data['legend']), 1),
                Y=np.array(self.metric_data['Y']),
                opts={
                    'title': self.name + ' metric over time',
                    'legend': self.metric_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id + 1)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    def print_current_losses_metrics(self, epoch, i, losses, metrics, t, t_data):
        if i:  # iter is not given in validation (confusing?)
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        else:
            message = '(epoch: %d, validation) ' % (epoch)
        for k, v in (OrderedDict(losses, **metrics)).items():  # not displayed in correct order in python <3.6
            if not i:
                k = '_'.join(k.split('_')[0:-1])
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # metrics: same format as |metrics| of plot_current_metrics




