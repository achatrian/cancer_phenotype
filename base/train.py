import time
import os
from options.train_options import TrainOptions
from data import create_dataset, create_dataloader
from models import create_model
from deploy import create_deployer
from utils import create_visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train_dataset = create_dataset(opt)
    train_dataset.setup()  # TEMP: for datasets that require a longer set-up time, this step is not done when making options
    train_dataloader = create_dataloader(train_dataset)
    if opt.val_epoch_freq:
        val_dataset = create_dataset(opt, validation_phase=True)
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset)
    print('task: {}, #training images = {:d}'.format(opt.task, len(train_dataset)))

    model = create_model(opt)
    model.setup(dataset=train_dataset)  # dataset is used by some models
    visualizer = create_visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.nepoch + opt.nepoch_decay):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.display_freq == 0 or total_steps % opt.print_freq == 0:
                model.evaluate_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), model.get_visual_paths(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                metrics = model.get_current_metrics()  # added by me
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses_metrics(epoch, epoch_iter, losses, metrics, t, t_data)
                if opt.display_id > 0:
                    epoch_progress = float(epoch_iter) / (len(train_dataloader) * opt.batch_size)
                    visualizer.plot_current_losses_metrics(epoch, epoch_progress, losses, metrics)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        if opt.val_epoch_freq and epoch % opt.val_epoch_freq == 0:
            val_start_time = time.time()
            with model.start_validation() as update_validation_meters:
                if opt.eval:
                    model.eval()
                for j, data in enumerate(val_dataloader):
                    val_start_time = time.time()
                    model.set_input(data)
                    model.test()
                    model.evaluate_parameters()
                    update_validation_meters()
            visualizer.reset()
            visualizer.display_current_results(model.get_current_visuals(), model.get_visual_paths(), epoch, True)
            losses_val = model.get_current_losses()
            metrics_val = model.get_current_metrics()
            visualizer.print_current_losses_metrics(epoch, None, losses_val, metrics_val, None, None)
            if opt.display_id > 0:
                visualizer.plot_current_losses_metrics(epoch, epoch_progress + 0.001, losses_val, metrics_val)
            print("Validated parameters at epoch {:d} \t Time Taken: {:d} sec".format(epoch, int(time.time() - val_start_time)))