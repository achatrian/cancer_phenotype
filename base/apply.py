from queue import Empty
import resource
import torch.multiprocessing as mp
from options.apply_options import ApplyOptions
from models import create_model
from datasets import create_dataset, create_dataloader
from deploy import create_deployer


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    opt.display_id = -1   # no visdom display
    model = create_model(opt)
    # model.setup()  # done in predictor process, as it fails when net is push to cuda otherwise
    if model:
        model.share_memory()
    dataset = create_dataset(opt)
    if opt.make_subset:
        dataset.make_subset()
    dataset.setup()  # NB swapped in position .make_subset() and .setup()
    dataloader, deployer = create_dataloader(dataset), create_deployer(opt)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
    resource.setrlimit(resource.RLIMIT_NOFILE, (3000, rlimit[1]))
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
    deployer.cleanup()





