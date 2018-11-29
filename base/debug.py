from options.apply_options import ApplyOptions
from models import create_model
from deploy import create_deployer


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    model = create_model(opt)
    model.setup()