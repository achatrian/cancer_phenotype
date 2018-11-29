from options.apply_options import ApplyOptions
from models import create_model
from deploy import create_deployer


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    model = create_model(opt)
    model.setup()
    model.share_memory()

    deployer = create_deployer(opt)
    deployer.setup()
    workers = deployer.get_workers()

    for i, worker in enumerate(workers):
        worker.start()
        print("Worker {} was started ...".format(i))

    for i, worker in enumerate(workers):
        print("... worker {} joined main.".format(i))





