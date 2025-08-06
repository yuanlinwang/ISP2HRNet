import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
        mismatched_keys = [k for k in model.state_dict() if k not in model_spec['sd']]
        print("Mismatched keys in the checkpoint:", mismatched_keys)
        mismatched_keys = [k for k in model_spec['sd'] if k not in model.state_dict()]
        print("Mismatched keys in the model:", mismatched_keys)
    return model
