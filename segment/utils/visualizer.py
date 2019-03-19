import os
import numpy as np
from base.utils.base_visualizer import BaseVisualizer, VisdomExceptionBase
from base.utils import utils, html


class SegmentVisualizer(BaseVisualizer):

    def __init__(self, opt):
        super().__init__(opt)

    def display_current_results(self, visuals, visuals_paths, epoch, save_result):
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
                label_html, label_html_row, path_html, path_html_row = '', '', '', ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    path = os.path.basename(visuals_paths[label.split('_')[0]][0])
                    image_numpy = utils.tensor2im(image[0, ...], label.endswith("_map"))
                    label_html_row += f'<td>{label}</td>'
                    path_html_row += f'<td>{path}</td>'
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += f'<tr>{label_html_row}</tr>'
                        label_html_row = ''
                        path_html += f'<tr>{path_html_row}</tr>'
                        path_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    path_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += f'<tr>{label_html_row}</tr>'
                    path_html += f'<tr>{path_html_row}</tr>'
                # pane col = image row
                assert all([images[0].shape == image.shape for image in images]);
                "Ensure all images have same shape"
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 2,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = f'<table>{label_html}{path_html}</table>'
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
            losses_legend = list(losses.keys()) + [loss + "_val" for loss in losses.keys() if
                                                   not loss.endswith('_val')]
            self.loss_data = {'X': [epoch - 0.05], 'Y': [[0] * len(losses_legend)], 'legend': losses_legend}
        self.loss_data['X'].append(epoch + epoch_progress)
        # fill with latest value if loss is not given for update
        self.loss_data['Y'].append([losses[k] if k in losses.keys() else self.loss_data['Y'][-1][i]
                                    for i, k in enumerate(self.loss_data['legend'])])
        if not hasattr(self, 'metric_data'):
            metrics_legend = list(metrics.keys()) + [metric + "_val" for metric in metrics.keys()]
            self.metric_data = {'X': [epoch - 0.05], 'Y': [[0] * len(metrics_legend)], 'legend': metrics_legend}
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
