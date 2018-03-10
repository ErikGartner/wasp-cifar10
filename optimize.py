import pickle

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope


def train_model_hyperopt(args):
    print('Calling train_model with: %s' % args)

    # Import keras in the training thread
    from train import train_model

    # Apply all the kwargs to the training function
    return train_model(**args)


def main():
    space = {
        'max_epochs': 50,
        'growth_rate': scope.int(hp.quniform('growth_rate', 12, 48, 12)),
        'batch_size': hp.choice('batch_size', [64, 84]),
        'dense_layers': hp.choice('dense_layers',
            [[6, 12, 24, 14], [6, 12, 32, 24]]),
        'compression': hp.quniform('compression', 0.4, 0.6, 0.1),
        'dropout': hp.quniform('dropout', 0, 0.5, 0.25),
    }

    trials = Trials()
    best = fmin(train_model_hyperopt,
                space=space,
                algo=tpe.suggest,
                max_evals=5,
                trials=trials)

    pickle.dump(trials, open('trials_object.p', 'wb'))
    print(best)


if __name__ == '__main__':
    main()
