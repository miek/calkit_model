import argparse
from keysight import *
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skrf

class Params:
    def model(self, freq, x):
        return self._model_func(freq, *self.apply_units(x))

    def initial_guess(self):
        return self._initial_guess

    def bounds(self):
        return self._bounds

    def apply_units(self, x):
        return x * self._units

class ShortParams(Params):
    def __init__(self, args):
        self._model_func = keysight_calkit_short

        self._initial_guess = [
            40,
            3.21e9,
            0,
            0,
            0,
            0,
            50,
        ]

        self._bounds = [
            (0, 1000),
            (0, 1e12),
            (-1e5, 1e5),
            (-1e5, 1e5),
            (-1e5, 1e5),
            (-1e5, 1e5),
            (40, 60),
        ]

        self._units = np.array([
            1e-12,
            1,
            1e-12,
            1e-24,
            1e-33,
            1e-42,
            1,
        ])

    def scpi_commands(self, prefix, x):
        ret = [prefix + ":TYPE SHORT"]
        params = [
            'DEL',
            'LOSS',
            'L0',
            'L1',
            'L2',
            'L3',
            'Z0',
        ]
        for i in range(len(x)):
            ret.append(f"{prefix}:{params[i]} {x[i]}")
        return ret


class OpenParams(Params):
    def __init__(self, args):
        self._model_func = keysight_calkit_open

        self._initial_guess = [
            40,
            3.21e9,
            0,
            0,
            0,
            0,
            50,
        ]

        self._bounds = [
            (0, 1000),
            (0, 1e12),
            (-10000, 10000),
            (-10000, 10000),
            (-10000, 10000),
            (-10000, 10000),
            (1, 1000),
        ]

        self._units = np.array([
            1e-12,
            1,
            1e-15,
            1e-27,
            1e-36,
            1e-45,
            1,
        ])

    def scpi_commands(self, prefix, x):
        ret = [prefix + ":TYPE OPEN"]
        params = [
            'DEL',
            'LOSS',
            'C0',
            'C1',
            'C2',
            'C3',
            'Z0',
        ]
        for i in range(len(x)):
            ret.append(f"{prefix}:{params[i]} {x[i]}")
        return ret


class LoadParams(Params):
    def __init__(self, args):
        self._model_func = keysight_calkit_load

        self._initial_guess = [
            40,
            3.21e9,
            50,
        ]

        self._bounds = [
            (0, 1000),
            (0, 1e12),
            (1, 1000),
        ]

        self._units = np.array([
            1e-12,
            1,
            1,
        ])

    def scpi_commands(self, prefix, x):
        ret = [prefix + ":TYPE LOAD"]
        params = [
            'DEL',
            'LOSS',
            'Z0',
        ]
        for i in range(len(x)):
            ret.append(f"{prefix}:{params[i]} {x[i]}")
        return ret


def error_complex(n1, n2):
    return n1.s[:,0,0] - n2.s[:,0,0]

def fit(x, standard=None, params=None):
    #print(f"{x=}")
    network = params.model(standard.frequency, x)
    residuals = error_complex(standard, network)
    return np.absolute(residuals)

def fit_scalar(x, standard=None, params=None):
    return np.sum(fit(x, standard, params))

def plot_results(measured_standard, modelled_guess, modelled_standard):
    smith_style = {
        'marker': '.',
        'markersize': 2,
        'linestyle': '',
    }

    plt.subplot(2, 2, 1)
    plt.title("Initial guess (S11)")
    modelled_guess.plot_s_smith(color='red', label='Modelled', **smith_style)
    measured_standard.plot_s_smith(color='blue', label='Measured', **smith_style)
    plt.legend(bbox_to_anchor=(0.73, 1), loc='upper left', borderaxespad=0)

    plt.subplot(2, 2, 2)
    plt.title("After optimisation (S11)")
    modelled_standard.plot_s_smith(color='red', label='Modelled', **smith_style)
    measured_standard.plot_s_smith(color='blue', label='Measured', **smith_style)
    plt.legend(bbox_to_anchor=(0.73, 1), loc='upper left', borderaxespad=0)

    plt.subplot(2, 2, 3)
    plt.title("Residual error")
    error = error_complex(measured_standard, modelled_guess)
    plt.scatter(error.real, error.imag, s=1)

    plt.subplot(2, 2, 4)
    plt.title("Residual error")
    error = error_complex(measured_standard, modelled_standard)
    plt.scatter(error.real, error.imag, s=1)

    plt.show()

def minimize(params, measured_standard):
    result = scipy.optimize.minimize(
        fit_scalar,
        x0=params.initial_guess(),
        bounds=params.bounds(),
        method='Nelder-Mead',
        #options={'maxiter': 3000},
        args=(measured_standard,params,)
    )
    print(result)
    return result.x

def least_squares(params, measured_standard):
    result = scipy.optimize.least_squares(
        fit,
        x0=params.initial_guess(),
        bounds=(
            [x[0] for x in params.bounds()],
            [x[1] for x in params.bounds()],
        ),
        #xtol=None,
        kwargs={'standard': measured_standard, 'params': params},
    )
    print(result)
    return result.x

parser = argparse.ArgumentParser(description='Derive calibration kit definitions from S-parameter measurements.')
parser.add_argument('type', choices=['short', 'open', 'load'])
parser.add_argument('file', type=open)
parser.add_argument('-m', '--method', choices=['least_squares', 'minimize'], default='least_squares')
parser.add_argument('-n', type=int)
args = parser.parse_args()

if args.type == 'short':
    params = ShortParams(args)
elif args.type == 'open':
    params = OpenParams(args)
elif args.type == 'load':
    params = LoadParams(args)
else:
    raise NotImplementedError

standard = skrf.Network(args.file)['0.0003-3ghz'] # TODO: add an argument to slice frequency range

# hack: skip the first few measurements, as they're a bit off for some reason
standard = standard[10:]

if args.method == 'least_squares':
    result = least_squares(params, standard)
elif args.method == 'minimize':
    result = minimize(params, standard)
else:
    raise ValueError

guess = params.model(standard.frequency, params.initial_guess())
model = params.model(standard.frequency, result)

# If `-n` is given, print SCPI commands to load the parameters.
sens = 1
stan = args.n
if not stan is None:
    result[0] *= 1e-12
    for cmd in params.scpi_commands(f"SENS{sens}:CORR:COLL:CKIT:STAN{stan}", result):
        print(cmd)
    print(f"SENS{sens}:CORR:COLL:METH:{args.type} 1")
    print(f"SENS{sens}:CORR:COLL:{args.type} 1")
    print("*WAI")
    print("SENS1:CORR:COLL:SAVE")

plot_results(standard, guess, model)
