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

    with deployer.queue_env() as output_queue:
        with deployer.queue_env() as input_queue:
            workers = deployer.get_workers(model, input_queue)
            for i, worker in enumerate(workers):
                worker.start()
                print("Worker {} is running ...".format(i))

            nimages = 0
            for j, data in enumerate(dataloader):
                nimages += data['input'].shape[0]
                data['idx'] = j
                input_queue.put(data)
                if j % opt.print_freq == 0:
                    print("Loaded: {} images".format(nimages))

            dataloader.gather(output_queue)






