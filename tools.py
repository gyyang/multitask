"""Utility functions."""

import os
import errno
import six
import json
import pickle
import numpy as np


def gen_feed_dict(model, trial, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: trial.x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = trial.x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                trial.x[:, i, :hp['rule_start']]

        feed_dict = {model.x: x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
    else:
        raise ValueError()

    return feed_dict


def _contain_model_file(model_dir):
    """Check if the directory contains model files."""
    for f in os.listdir(model_dir):
        if 'model.ckpt' in f:
            return True
    return False


def _valid_model_dirs(root_dir):
    """Get valid model directories given a root directory."""
    return [x[0] for x in os.walk(root_dir) if _contain_model_file(x[0])]


def valid_model_dirs(root_dir):
    """Get valid model directories given a root directory(s).

    Args:
        root_dir: str or list of strings
    """
    if isinstance(root_dir, six.string_types):
        return _valid_model_dirs(root_dir)
    else:
        model_dirs = list()
        for d in root_dir:
            model_dirs.extend(_valid_model_dirs(d))
        return model_dirs


def load_log(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def save_log(log): 
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['rng'] = np.random.RandomState(hp['seed']+1000)
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def find_all_models(root_dir, hp_target):
    """Find all models that satisfy hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters

    Returns:
        model_dirs: list of model directories
    """
    dirs = valid_model_dirs(root_dir)

    model_dirs = list()
    for d in dirs:
        hp = load_hp(d)
        if all(hp[key] == val for key, val in hp_target.items()):
            model_dirs.append(d)

    return model_dirs


def find_model(root_dir, hp_target, perf_min=None):
    """Find one model that satisfies hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters
        perf_min: float or None. If not None, minimum performance to be chosen

    Returns:
        d: model directory
    """
    model_dirs = find_all_models(root_dir, hp_target)
    if perf_min is not None:
        model_dirs = select_by_perf(model_dirs, perf_min)

    if not model_dirs:
        # If list empty
        print('Model not found')
        return None, None

    d = model_dirs[0]
    hp = load_hp(d)

    log = load_log(d)
    # check if performance exceeds target
    if log['perf_min'][-1] < hp['target_perf']:
        print("""Warning: this network perform {:0.2f}, not reaching target
              performance {:0.2f}.""".format(
              log['perf_min'][-1], hp['target_perf']))

    return d


def select_by_perf(model_dirs, perf_min):
    """Select a list of models by a performance threshold."""
    new_model_dirs = list()
    for model_dir in model_dirs:
        log = load_log(model_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > perf_min:
            new_model_dirs.append(model_dir)
    return new_model_dirs


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H
