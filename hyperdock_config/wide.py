import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.pyll.stochastic


SPACE = hp.choice('network_layout', [
    {
        'max_epochs': 100,

        # Big network, near memory limit
        'dense_layers': [20, 20, 20],
        'batch_size': 32,

        # Params
        'growth_rate': scope.int(hp.quniform('growth_rate', 40, 60, 5)),
        'start_lr': 2 ** scope.int(hp.quniform('start_lr', -6, 1, 1)),
        'dropout': hp.quniform('dropout', 0, 0.2, 0.05),
    }
])


if __name__ == '__main__':
    # Code that loads the space and samples from it as a test.
    print(hyperopt.pyll.stochastic.sample(SPACE))

# max_epochs=300, start_lr=0.1,
# dense_layers=[20, 20, 20], growth_rate=60, compression=0.5,
# dropout=0.0, weight_decay=1e-4, batch_size=64, logdir='./logs',
# weightsdir='./weights', lr_decrease_factor=0.5, lr_patience=10,
# nbr_gpus=1
