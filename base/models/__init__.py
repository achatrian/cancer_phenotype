import importlib
from base.models.base_model import BaseModel


def find_model_using_name(model_name, task_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    task_module = importlib.import_module(task_name)
    model_filename = task_name + ".models." + model_name.lower() + "_model"
    modellib = importlib.import_module(model_filename, package=task_module)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and any(cls0.__module__.endswith(BaseModel.__module__) for cls0 in cls.__mro__):  # check that base class is BaseModel
            # issubclass doesn't work if import was absolute in model definition, as here it is relative
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))

    return model


def get_option_setter(model_name, task_name):
    model_class = find_model_using_name(model_name, task_name)
    return model_class.modify_commandline_options


def create_model(opt):
    if opt.model and opt.model.lower() != 'none':
        model = find_model_using_name(opt.model, opt.task)
        instance = model(opt)
        print("model [%s] was created" % (instance.name()))
    else:
        instance = None
        print("no model was created")
    return instance
