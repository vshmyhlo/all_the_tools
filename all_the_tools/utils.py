import random

import numpy as np


def seed_python(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.set_random_seed(seed)
    np.random.seed(seed)
