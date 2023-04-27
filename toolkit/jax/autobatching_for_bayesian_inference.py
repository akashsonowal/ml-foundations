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
  result += jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=10.))
  result += jnp.sum(-jnp.log(1 + jnp.exp(-(2*y - 1) * jnp.dot(all_x, beta))))
  return result

batched_log_joint = jax.jit(jax.vmap(log_joint))

# define elbo and its gradient
def elbo(beta_loc, beta_log_scale, epsilon):
  beta_sample = beta_loc + jnp.exp(beta_log_scale) * epsilon
  return jnp.mean(batched_log_joint(beta_sample), 0) + jnp.sum(beta_log_scale - 0.5 * np.log(2*np.pi)) # mean of joint prob + penalty term to avoid high vraiance

elbo = jax.jit(elbo)
elbo_val_and_grad = jax.jit(jax.value_and_grad(elbo, argnums=(0, 1)) # for differentiation w.r.t beta_loc and beta_log_scale

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
  elbo_val, (beta_loc_grad, beta_log_scale_grad) = elbo_val_and_grad(beta_loc, beta_log_scale_grad, epsilon)
  beta_loc += step_size * beta_loc_grad
  beta_log_scale += step_size *  beta_log_scale_grad
  if i % 10 == 0:
    print(f'{i}\t{elbo_val})
          
 # Visualize results
figure(figsize=(7, 7))
plot(true_beta, beta_loc, '.', 'Approximated Posterior Means')
plot(true_beta, beta_loc + 2*jnp.exp(beta_log_scale), 'r.', label='Approximated Posterior $2\sigma$ Error Bars')
plot(true_beta, beta_loc - 2*jnp.exp(beta_log_scale), 'r.')
plot_scale = 3
plot([-plot_scale, plot_scale], [-plot_scale, plot_scale], 'k')
xlabel("True beta")
ylabel("Estimated beta")
legend(loc="best")
