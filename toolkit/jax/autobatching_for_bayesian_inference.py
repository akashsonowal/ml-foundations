import functools
import itertools
import re
import sys
import time

from matplotlib.pyplot import *

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

import numpy as np
import scipy as sp

# Generate a fake classification dataset
np.random.seed(100009)

num_features = 10
num_points = 100

true_beta = np.random.randn(num_features).astype(jnp.float32)
all_x = np.random.randn(num_points, num_features).astype(jnp.float32)
y = (np.random.rand(num_points) < sp.special.expit(all_x.dot(true_beta))).astype(jnp.int32)

# log joint function
@jax.jit
def log_joint(beta):
  result = 0.
  return result

batched_log_joint = jax.jit(jax.vmap(log_joint))

# define elbo and its gradient
def elbo(beta_loc, beta_log_scale, epsilon):
  pass

def normal_sample(key, shape):
  "Convenient function for quasi-stateful RNG."
  new_key,sub_key = random.split(key)
  return new_key, random.normal(sub_key,shape)  

normal_sample = jax.jit(normal_sample, static_args=(1,))
key = random.PRNGKey(10003)

# priors
beta_loc = jnp.zeros(num_features, jnp.float32) # mean: beta_loc
beta_log_scale = jnp.zeros(num_features, jnp.float32) # std dev: exp(beta_log_scale)

step_size = 0.01
batch_size = 128

epsilon_shape = (batch_size, features) # random noise to avoid local minima
for i in range(1000):
  key, epsilon = normal_sample(key, epsilon_shape)
  pass
