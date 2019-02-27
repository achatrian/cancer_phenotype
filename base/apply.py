from queue import Empty
import resource
import torch.multiprocessing as mp
from options.apply_options import ApplyOptions
from models import create_model
from data import create_dataset, create_dataloader
from deploy import create_deployer


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    model = create_model(opt)
    # model.setup()  # done in predictor process, as it fails when net is push to cuda otherwise
    if model:
        model.share_memory()
    dataset = create_dataset(opt)
    dataloader = create_dataloader(dataset)
    deployer = create_deployer(opt)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # https://github.com/fastai/fastai/issues/23#issuecomment-345091054
    resource.setrlimit(resource.RLIMIT_NOFILE, (3000, rlimit[1]))
    barrier = mp.Barrier(2)  # ensure main doesn't stop before gatherer
    parent_conn, child_conn = mp.Pipe()
    with deployer.queue_env(sentinel=False) as output_queue:
        with deployer.queue_env(sentinel=True) as input_queue:
            if opt.ndeploy_workers:
                workers = deployer.get_workers(model, input_queue, output_queue, sync=(barrier, child_conn))
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
            else:
                # run in main process
                workers = []
                nimages = 0
                for j, data in enumerate(dataloader):
                    nimages += data['input'].shape[0]
                    data['idx'] = j
                    input_queue.put(data)
                    input_queue.put(None)
                    deployer.run_worker(0, opt, model, input_queue, output_queue)
                try:
                    while True:
                        _ = input_queue.get_nowait()
                        input_queue.task_done()
                except Empty:
                    pass
                print("Processing terminated")
                deployer.gather(deployer, output_queue)  # don't wait for input queue to join to do this
        if workers:
            for worker in workers:
                if worker.name.endswith('Worker'):
                    worker.join()  # join before joining output queue

    if workers and workers[-1].name.endswith('Gatherer'):
        workers[-1].join()





