import yaml


def customConfigs(args, config):
    for key in args:
        if key == 'task' or key == 'config' or key == 'facility' or key == 'queue' \
            or key == 'experiment_name' or key == 'n_cores' or key == 'run_number':
            pass
        elif args[key] == None:
            pass
        else:
            try:
                # Using '+' as the seperator, please make changes as you'd like
                k1, k2 = key.split('+')  #TODO: Generalize for arbitrary layers
                config[k1][k2] = args[key]
            except Exception as e:
                pass
    return config


def editConfig(args):
    config_filepath = args['config_file']
    new_config_filepath = config_filepath[:-5] + '-tmp' + '.yaml'

    with open(config_filepath, "r") as config_file:
        config = yaml.safe_load(config_file)
        config = customConfigs(args, config)

    with open(new_config_filepath, 'w') as out:
        yaml.dump(config, out)

    print('Config edited according to the inputs.')
