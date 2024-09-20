import jax.numpy as jnp
from jax.nn import tanh, sigmoid, relu
from jax.experimental.ode import odeint
from jax import lax, jit, vmap, pmap, jacfwd, jacrev
from jax.lax import dynamic_slice
from functools import partial


# define system of equations
@jit
def system(s, t, params, s_cap):

    # unpack params
    r, A = params

    # growth rates
    dsdt = s * (r + A @ s) * (1. - s / s_cap)

    return dsdt


@partial(jit, static_argnums=(0,))
def runODE(shapes, tf, s, s_cap, z):
    # transform and reshape parameters
    params = reshape(shapes, z)

    # get initial condition of species and mediators
    s_ic = s[0]

    # integrate
    t_eval = jnp.linspace(0., tf, 10)
    s_pred = odeint(system, s_ic, t_eval, params, s_cap,
                    rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)

    # return final time point
    return s_pred[-1]


# integrate batch
batchODE = pmap(runODE,
                in_axes=(None, 0, 0, 0, None,),
                static_broadcasted_argnums=(0,))


# gradient of solution of ODE w.r.t. parameters
@partial(jit, static_argnums=(0,))
def jacODE(shapes, tf, s, s_cap, z):
    return jacrev(runODE, -1)(shapes, tf, s, s_cap, z)


# evaluate Jacobian in batches
batch_jacODE = pmap(jacODE,
                    in_axes=(None, 0, 0, 0, 0, None, None, None),
                    static_broadcasted_argnums=(0,))


# evaluate ODE at specified time points
@partial(jit, static_argnums=(0,))
def runODE_teval(shapes, t_eval, s_ic, s_cap, z):
    # transform and reshape parameters
    params = reshape(shapes, z)

    # integrate
    return odeint(system, s_ic, jnp.array(t_eval), params, s_cap,
                  rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)


# reshape parameter vector into matrix/vectors
@partial(jit, static_argnums=(0,))
def reshape(shapes, params):
    return [jnp.reshape(dynamic_slice(params, (k1,), (k2 - k1,)), shape) for k1, k2, shape in
            zip(shapes[0], shapes[0][1:], shapes[1])]


# evaluate rmse
@partial(jit, static_argnums=(0,))
def root_mean_squared_error(shapes, tf, s, s_cap, z):

    # measured values
    true = s[-1]

    # integrate ode
    pred = jnp.nan_to_num(runODE(shapes, tf, s, s_cap, z))

    # error
    error = jnp.nan_to_num(true - pred)

    # root mean square error
    return jnp.sqrt(jnp.mean(error ** 2))


# gradient of rmse
@partial(jit, static_argnums=(0,))
def grad_root_mean_squared_error(shapes, tf, s, s_cap, z):
    return jacrev(root_mean_squared_error, -1)(shapes, tf, s, s_cap, z)
