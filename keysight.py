# Ref: [SOLT Calibration Standards Creation](https://scikit-rf.readthedocs.io/en/latest/examples/metrology/SOLT%20Calibration%20Standards%20Creation.html)

from skrf.media import DefinedGammaZ0
import numpy as np


def keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0):
    if offset_delay or offset_loss:
        alpha_l = (offset_loss * offset_delay) / (2 * offset_z0)
        alpha_l *= np.sqrt(freq.f / 1e9)
        beta_l = 2 * np.pi * freq.f * offset_delay + alpha_l
        zc = offset_z0 + (1 - 1j) * (offset_loss / (4 * np.pi * freq.f)) * np.sqrt(freq.f / 1e9)
        gamma_l = alpha_l + beta_l * 1j

        medium = DefinedGammaZ0(frequency=freq, z0=50, Z0=zc, gamma=gamma_l)
        offset_line = medium.line(d=1, unit='m', z0=medium.Z0, embed=True)
        return medium, offset_line
    else:
        medium = DefinedGammaZ0(frequency=freq, Z0=offset_z0)
        line = medium.line(d=0)
        return medium, line


def keysight_calkit_open(freq, offset_delay, offset_loss, c0, c1, c2, c3, offset_z0=50):
    medium, line = keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0)
    # Capacitance is defined with respect to the port impedance offset_z0, not the lossy
    # line impedance. In scikit-rf, the return values of `shunt_capacitor()` and `medium.open()`
    # methods are (correctly) referenced to the port impedance.
    if c0 or c1 or c2 or c3:
        poly = np.poly1d([c3, c2, c1, c0])
        capacitance = medium.shunt_capacitor(poly(freq.f)) ** medium.open()
    else:
        capacitance = medium.open()
    return line ** capacitance


def keysight_calkit_short(freq, offset_delay, offset_loss, l0, l1, l2, l3, offset_z0=50):
    # Inductance is defined with respect to the port impedance offset_z0, not the lossy
    # line impedance. In scikit-rf, the return values of `inductor()` and `medium.short()`
    # methods are (correctly) referenced to the port impedance.
    medium, line = keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0)
    if l0 or l1 or l2 or l3:
        poly = np.poly1d([l3, l2, l1, l0])
        inductance = medium.inductor(poly(freq.f)) ** medium.short()
    else:
        inductance = medium.short()
    return line ** inductance


def keysight_calkit_load(freq, offset_delay=0, offset_loss=0, offset_z0=50):
    medium, line = keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0)
    load = medium.match()
    return line ** load


def keysight_calkit_thru(freq, offset_delay=0, offset_loss=0, offset_z0=50):
    medium, line = keysight_calkit_offset_line(freq, offset_delay, offset_loss, offset_z0)
    thru = medium.thru()
    return line ** thru