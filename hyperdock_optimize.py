import json

from train import train_model


def read_params():
    with open('/hyperdock/params.json', 'r') as f:
        return json.load(f)


if __name__ == '__main__':

    params = read_params()
    params['dump_dir'] = '/hyperdock/out'
    params['logdir'] = '/hyperdock/out/logs'
    params['weightsdir'] = '/hyperdock/out'

    loss = train_model(**params)

    with open('/hyperdock/loss.json', 'w') as f:
        json.dump(loss, f)
