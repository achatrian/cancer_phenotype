from queue import Empty
import resource
import torch.multiprocessing as mp
from options.apply_options import ApplyOptions
from models import create_model
from data import create_dataset, create_dataloader
from deploy import create_deployer


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    opt.display_id = -1   # no visdom display
    model = create_model(opt)
    # model.setup()  # done in predictor process, as it fails when net is push to cuda otherwise
    if model:
        model.share_memory()
    dataset = create_dataset(opt)
    dataset.setup()
    if opt.make_subset:
        dataset.make_subset()
    dataloader = create_dataloader(dataset)
    deployer = create_deployer(opt)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
    resource.setrlimit(resource.RLIMIT_NOFILE, (3000, rlimit[1]))
    if __debug__ and not bool(opt.ndeploy_workers):
        # not running multiprocessing - for debug purposes
        with deployer.queue_env(sentinel=False) as output_queue:
            with deployer.queue_env(sentinel=True) as input_queue:
                for j, data in enumerate(dataloader):
                    input_queue.put(data)
                    if j >= 25:
                        break
                deployer.run_worker(-1, opt, model, input_queue, output_queue)
                output_queue.get()
    else:
        with deployer.queue_env(sentinel=False) as output_queue:
            with deployer.queue_env(sentinel=True) as input_queue:
                workers = deployer.get_workers(model, input_queue, output_queue, sync=())
                for i, worker in enumerate(workers):
                    worker.start()
                    print(f"Worker {i} - {worker.name} is running ...")
                nimages = 0
                for j, data in enumerate(dataloader):
                    nimages += data['input'].shape[0]
                    data['idx'] = j
                    input_queue.put(data)
                    if j % 10 == 0:
                        print("Loaded: {} images".format(nimages))
            if workers:
                for worker in workers:
                    if worker.name.endswith('Worker'):
                        worker.join()  # join before joining output queue

        if workers and workers[-1].name.endswith('Gatherer'):
            workers[-1].join()





