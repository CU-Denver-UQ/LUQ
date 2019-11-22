#!/usr/bin/env python

import numpy as np
from luq import LUQ
from setup_oscil import *

learn = LUQ(predicted_time_series, observed_time_series, times)

time_start_idx = 20
time_end_idx = 49

learn.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                 num_clean_obs=50, tol=1.0e-2, min_knots=5, max_knots=20)
learn.dynamics()
learn.learn_qois_and_transform()








