import os
import multiprocessing
from scipy.io import savemat, loadmat
from scipy.special import logsumexp, comb
import numpy as np
# from tqdm import tqdm

# import libraries to compute gradients
import jax
from jax import vjp, jacfwd, vmap, pmap, random
from jax.experimental.ode import odeint
from jax.nn import tanh, sigmoid, softmax, relu
from scipy.optimize import minimize, curve_fit

# system of equations
from .glv3_system import *

# curve fit to check convergence
from .utilities import *

class gLV:
    def __init__(self, dataframe, species, lmbda=0., s_cap=10., rng_seed=0):

        # lmbda is L1 regularization coefficient
        self.lmbda = lmbda

        # s_cap: carrying capacity of species
        self.s_cap = s_cap

        # set rng for parameter init
        self.rng_seed = int(rng_seed)

        # number of available devices for parallelizing code
        self.n_devices = 1  # jax.device_count()

        # dimensions
        self.n_s = len(species)

        # save names of species and mediators
        self.species = list(species)

        # set up data
        self.t, self.S = process_df(dataframe, species)

        # init params

        # growth rate of species
        r = 0.3 * np.ones(self.n_s)
        r_std = np.ones_like(r)

        # [A]_ij = rate that species j affects species i
        A = np.zeros([self.n_s, self.n_s])
        A_std = np.ones_like(A) / self.n_s

        # dimension of 3rd order basis
        dim1 = self.n_s
        dim2 = int(comb(self.n_s - 1, 2))

        # [B]_ijk = rate that species j * species k affects species i
        B = np.zeros([dim1, dim2])
        B_std = np.ones_like(B) / dim2

        # list of parameters
        params = [r, A, B]
        params_std = [r_std, A_std, B_std]

        # prior of zero promotes sparsity
        prior = [np.zeros_like(r), np.zeros_like(A), np.zeros_like(B)]

        # transform and flatten params
        z = jnp.concatenate([p.ravel() for p in params])
        z_std = jnp.concatenate([p.ravel() for p in params_std])

        # number of parameters
        self.d = len(z)
        y = np.random.randn(self.d)

        # randomly init parameters
        self.z = z + z_std * y

        # set parameter prior
        self.prior = jnp.concatenate([p.ravel() for p in prior])

        # determine shapes of parameters
        shapes = []
        k_params = []
        self.n_params = 0
        for param in params:
            shapes.append(param.shape)
            k_params.append(self.n_params)
            self.n_params += param.size
        k_params.append(self.n_params)

        # make shapes immutable
        self.shapes = tuple((tuple(k_params), tuple(shapes)))

    # compute residual between parameter estimate and prior
    def param_res(self, params):
        # residuals
        res = params - self.prior
        return res

    # mean squared error
    def rmse(self, z):

        # init mse
        rmse = 0.

        # total number of samples
        N = len(self.t)

        # evaluate samples in batches
        for idx in range(N):
            # gradient of log likelihood
            rmse += root_mean_squared_error(self.shapes,
                                            self.t[idx],
                                            self.S[idx],
                                            self.s_cap,
                                            z) / N

        # print("nlp", self.NLP)
        return rmse

    def fit_rmse(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=10000):

        # init moments of gradient
        m = np.zeros_like(self.z)
        v = np.zeros_like(self.z)
        t = 0
        epoch = 0

        # order of samples
        N = len(self.t)
        order = np.arange(N)

        # initialize function evaluations
        f = []

        while epoch <= epochs:
            if epoch % 10 == 0:
                # check convergence
                f.append(self.rmse(self.z))

                print("Epoch {:.0f}, RMSE: {:.5f}".format(epoch, f[-1]))
            epoch += 1

            # stochastic gradient descent
            np.random.shuffle(order)

            # take step for each sample
            for idx in order:

                # gradient of rmse w.r.t. parameters
                gradient = grad_root_mean_squared_error(self.shapes,
                                                        self.t[idx],
                                                        self.S[idx],
                                                        self.s_cap,
                                                        self.z)

                # add L1 weight decay
                gradient += self.lmbda * jnp.sign(self.param_res(self.z))

                # ignore exploding gradients
                gradient = np.where(np.abs(gradient) < 1000., gradient, 0.)

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # return list of function evaluations
        return f

    def predict_point(self, s_test, t_eval):

        # convert to arrays
        t_eval = np.array(t_eval, dtype=np.float32)

        # separate state
        s_test = np.atleast_2d(s_test)

        # make predictions given initial conditions and evaluation times
        s_out = runODE_teval(self.shapes, t_eval, s_test[0], self.s_cap, self.z)

        return s_out
