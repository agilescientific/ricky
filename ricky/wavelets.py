"""
Make wavelets.

Author: Matt Hall
Email: matt@agilescientific.com

Licence: Apache 2.0

Copyright 2022 Agile Scientific

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings

import numpy as np
import xarray as xr
import scipy.signal


def _wrap_xarray(t, f, w, attrs):
    """
    Wrap a wavelet as an xarray.DataArray.

    Args:
        t (array-like): The time vector.
        f (array-like): The frequency vector.
        w (array-like): The wavelet.
        attrs (dict): The attributes to add to the xarray.DataArray.

    Returns:
        xarray.DataArray. The wavelet.
    """
    nw, fw = f.shape  # nw: number of wavelets, fw: frequencies per wavelet

    f_ = xr.DataArray(np.squeeze(f, axis=1), dims=['frequency'], attrs={'units': 'Hz'})
    t_ = xr.DataArray(t, dims=['time'], attrs={'units': 's'})

    if nw == 1:
        w_ = xr.DataArray(w, dims=['time'], coords=[t_])
        attrs.update({'frequency': f_})
    elif f_.shape[-1] == 1:
        w_ = xr.DataArray(w, name='amplitude', dims=['frequency', 'time'], coords=[f_, t_])
    w_ = w_.assign_attrs(attrs)
    return w_


def _get_time(duration, dt, sym=True):
    """
    Make a time vector.

    If `sym` is `True`, the time vector will have an odd number of samples,
    and will be symmetric about 0. If it's False, and the number of samples
    is even (e.g. duration = 0.016, dt = 0.004), then 0 will bot be center.
    """
    # This business is to avoid some of the issues with `np.arange`:
    # (1) unpredictable length and (2) floating point weirdness, like
    # 1.234e-17 instead of 0. Not using `linspace` because figuring out
    # the length and offset gave me even more of a headache than this.
    n = int(duration / dt)
    odd = n % 2
    k = int(10**-np.floor(np.log10(dt)))
    dti = int(k * dt)  # integer dt

    if (odd and sym):
        t = np.arange(n)
    elif (not odd and sym):
        t = np.arange(n + 1)
    elif (odd and not sym):
        t = np.arange(n)
    elif (not odd and not sym):
        t = np.arange(n) - 1

    t -= t[-1] // 2

    return dti * t / k


def _generic(func, duration, dt, f, t=None, return_t=True, taper='blackman', sym=True, name=None):
    """
    Generic wavelet generator: applies a window to a continuous function.

    Args:
        func (function): The continuous function, taking t, f as arguments.
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments. If `t`
            is not a reasonably well- and regularly sampled array, you should
            probably not use a taper.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none' or None. To apply your own function, pass a function taking
            only the length of the window and returning the window function.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.
        name (str): The name of the wavelet; added to the attribute dict.

    Returns:
        ndarray. wavelet(s) with centre frequency f sampled on t. If you
            passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if not return_t:
        m = "return_t is deprecated. In future releases, return_t will always be True."
        warnings.warn(m, DeprecationWarning, stacklevel=2)

    f = np.asanyarray(f).reshape(-1, 1)

    # Compute time domain response.
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)
        t = np.array(t)

    t[t == 0] = 1e-12  # Avoid division by zero.
    f[f == 0] = 1e-12  # Avoid division by zero.

    w = func(t, f)

    if taper is not None:
        tapers = {
            'bartlett': np.bartlett,
            'blackman': np.blackman,
            'hamming': np.hamming,
            'hanning': np.hanning,
            'none': lambda _: 1,
        }
        taper = tapers.get(taper, taper)
        w *= taper(t.size)

    attrs = {
        'kind': name or '',
        'taper': taper,
        'frequency': np.squeeze(f, axis=1),
        }
    return _wrap_xarray(t, f, w, attrs)


def sinc(duration, dt, f, t=None, return_t=True, taper='blackman', sym=True):
    """
    sinc function centered on t=0, with a dominant frequency of f Hz.

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w, t = bruges.filters.sinc(0.256, 0.002, 40)
        plt.plot(t, w)

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.

    Returns:
        ndarray. sinc wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t = True` then a tuple of (wavelet, t) is returned.
    """
    def func(t_, f_):
        return np.sin(2*np.pi*f_*t_) / (2*np.pi*f_*t_)

    return _generic(func, duration, dt, f, t, return_t, taper, sym=sym)


def cosine(duration, dt, f, t=None, return_t=True, taper='gaussian', sigma=None, sym=True, name='sinc'):
    """
    With the default Gaussian window, equivalent to a 'modified Morlet'
    also sometimes called a 'Gabor' wavelet. The `bruges.filters.gabor`
    function returns a similar shape, but with a higher mean frequancy,
    somewhere between a Ricker and a cosine (pure tone).

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w, t = bruges.filters.cosine(0.256, 0.002, 40)
        plt.plot(t, w)

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.
        sigma (float): Width of the default Gaussian window, in seconds.
            Defaults to 1/8 of the duration.

    Returns:
        ndarray. sinc wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    if sigma is None:
        sigma = duration / 8

    def func(t_, f_):
        return np.cos(2 * np.pi * f_ * t_)

    def taper(length):
        return scipy.signal.gaussian(length, sigma/dt)

    return _generic(func, duration, dt, f, t, return_t, taper, sym=sym, name='cosine')


def gabor(duration, dt, f, t=None, return_t=True, sym=True):
    """
    Generates a Gabor wavelet with a peak frequency f0 at time t.

    https://en.wikipedia.org/wiki/Gabor_wavelet

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w, t = bruges.filters.gabor(0.256, 0.002, 40)
        plt.plot(t, w)

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.

    Returns:
        ndarray. Gabor wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.
    """
    def func(t_, f_):
        return np.exp(-2 * f_**2 * t_**2) * np.cos(2 * np.pi * f_ * t_)

    return _generic(func, duration, dt, f, t, sym=sym, name='gabor')


def ricker(duration, dt, f, t=None, sym=True):
    r"""
    Also known as the mexican hat wavelet, models the function:

    .. math::
        A =  (1 - 2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w, t = bruges.filters.ricker(0.256, 0.002, 40)
        plt.plot(t, w)

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        ndarray. Ricker wavelet(s) with centre frequency f sampled on t. If
            you passed `return_t=True` then a tuple of (wavelet, t) is returned.

    """
    f = np.asanyarray(f).reshape(-1, 1)

    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)

    pft2 = (np.pi * f * t)**2
    w = (1 - (2 * pft2)) * np.exp(-pft2)
    
    attrs = {'kind': 'ricker', 'frequency': np.squeeze(f, axis=1)}
    return _wrap_xarray(t, f, w, attrs)
