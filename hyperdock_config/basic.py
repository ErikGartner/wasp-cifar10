import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope


SPACE = {
    'max_epochs': 50,
    'growth_rate': scope.int(hp.quniform('growth_rate', 40, 72, 12)),
    'compression': hp.quniform('compression', 0.4, 0.6, 0.1),
    'dropout': hp.quniform('dropout', 0, 0.2, 0.1),
}
