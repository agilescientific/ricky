"""
Make wavelets.

Author: Matt Hall
Email: matt@agilescientific.com
Licence: Apache 2.0
"""
import warnings

import numpy as np
from numpy.typing import NDArray, ArrayLike
import xarray as xr
import scipy.signal


def _wrap_xarray(t: NDArray, f: NDArray, w: NDArray, attrs: dict) -> xr.DataArray:
    """
    Wrap a wavelet as an xarray.DataArray.

    Args:
        t (ndarray): The time vector.
        f (ndarray): The frequency vector.
        w (ndarray): The wavelet.
        attrs (dict): The attributes to add to the xarray.DataArray.

    Returns:
        xarray.DataArray. The wavelet.
    """
    f_ = xr.DataArray(np.squeeze(f, axis=1), dims=['frequency'], attrs={'units': 'Hz'})
    t_ = xr.DataArray(t, dims=['time'], attrs={'units': 's'})

    w_ = xr.DataArray(w, name='amplitude', dims=['frequency', 'time'], coords=[f_, t_])
    w_ = w_.assign_attrs(attrs)
    return w_


def _get_time(duration: float, dt: float, sym: bool=True) -> NDArray:
    """
    Make a time vector.

    If `sym` is `True`, the time vector will have an odd number of samples,
    and will be symmetric about 0. If it's `False`, and the number of samples
    is even (e.g. duration = 0.016, dt = 0.004), then 0 will not be center.

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        sym (bool): If True then the wavelet is forced to have an odd number of
            samples and the central sample is at 0 time.

    Returns:
        ndarray: The time vector.
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


def _generic(func, duration, dt, f, t=None, taper='blackman', sym=True, name=None):
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
        xarray.DataArray: Wavelet(s) with centre frequency f sampled on t.
    """
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
        taper_func = tapers.get(taper, taper)
        w *= taper_func(t.size)

    attrs = {
        'kind': name or '',
        'taper': taper,
        'frequency': np.squeeze(f, axis=1),
        }

    return _wrap_xarray(t, f, w, attrs)


def sinc(duration, dt, f, t=None, taper='blackman', sym=True):
    """
    Sinc function centered on t=0, with a dominant frequency of f Hz.

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
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.

    Returns:
        xarray.DataArray: Sinc wavelet(s) with centre frequency f sampled on t.
    """
    def func(t_, f_):
        return np.sin(2*np.pi*f_*t_) / (2*np.pi*f_*t_)

    return _generic(func, duration, dt, f, t, taper, sym=sym)


def cosine(duration, dt, f, t=None, taper='gaussian', sigma=None, sym=True):
    """
    With the default Gaussian window, equivalent to a 'modified Morlet'
    also sometimes called a 'Gabor' wavelet. The `bruges.filters.gabor`
    function returns a similar shape, but with a higher mean frequancy,
    somewhere between a Ricker and a cosine (pure tone).

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w = ricky.cosine(duration=0.256, dt=0.002, f=40)
        w.plot()

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Dominant frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        taper (str or function): The window or tapering function to apply.
            To use one of NumPy's functions, pass 'bartlett', 'blackman' (the
            default), 'hamming', or 'hanning'; to apply no tapering, pass
            'none'. To apply your own function, pass a function taking only
            the length of the window and returning the window function.
        sigma (float): Width of the default Gaussian window, in seconds.
            Defaults to 1/8 of the duration.
        sym (bool): If True then the wavelet is forced to have an odd number of
            samples and the central sample is at 0 time.

    Returns:
        xarray.DataArray: Cosine wavelet(s) with centre frequency f sampled on
            t.
    """
    if sigma is None:
        sigma = duration / 8

    def func(t_, f_):
        return np.cos(2 * np.pi * f_ * t_)

    def taper(length):
        return scipy.signal.gaussian(length, sigma/dt)

    return _generic(func, duration, dt, f, t, taper, sym=sym, name='cosine')


def gabor(duration: float, dt: float, f: ArrayLike, t: ArrayLike=None, sym: bool=True) -> xr.DataArray:
    """
    Generates a Gabor wavelet with a peak frequency f0 at time t.

    https://en.wikipedia.org/wiki/Gabor_wavelet

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w = ricky.gabor(duration=0.256, dt=0.002, f=40)
        w.plot()

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        sym (bool): If True then the wavelet is forced to have an odd number of
            samples and the central sample is at 0 time.

    Returns:
        xarray.DataArray: Gabor wavelet(s) with centre frequency f sampled on t.
    """
    def func(t_, f_):
        return np.exp(-2 * f_**2 * t_**2) * np.cos(2 * np.pi * f_ * t_)

    return _generic(func, duration, dt, f, t, sym=sym, name='gabor')


def ricker(duration: float, dt: float, f: ArrayLike, t: ArrayLike=None, sym: bool=True) -> xr.DataArray:
    r"""
    Also known as the mexican hat wavelet, models the function:

    .. math::
        A =  (1 - 2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w = ricky.ricker(duration=0.256, dt=0.002, f=40)
        w.plot()

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        sym (bool): If True then the wavelet is forced to have an odd number of
            samples and the central sample is at 0 time.

    Returns:
        xarray.DataArray: Ricker wavelet(s) with centre frequency f sampled on
            t.
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


def berlage(duration, dt, f, n=2, alpha=180, phi=-np.pi/2, t=None, sym=True):
    r"""
    Generates a Berlage wavelet with a peak frequency f. Implements

    .. math::

        w(t) = AH(t) t^n \mathrm{e}^{- \alpha t} \cos(2 \pi f_0 t + \phi_0)

    as described in Aldridge, DF (1990), The Berlage wavelet, GEOPHYSICS
    55 (11), p 1508-1511. Berlage wavelets are causal, minimum phase and
    useful for modeling marine airgun sources.

    If you pass a 1D array of frequencies, you get a wavelet bank in return.

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w = ricky.berlage(duration=0.256, dt=0.002, f=40)
        plt.plot(t, w)

    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (array-like): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        n (float): The time exponent; non-negative and real.
        alpha(float): The exponential decay factor; non-negative and real.
        phi (float): The phase.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        xarray.DataArray: Berlage wavelet(s) with centre frequency f sampled on
            t.
    """
    f = np.asanyarray(f).reshape(-1, 1)
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)

    H = np.heaviside(t, 0)
    w = H * t**n * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t + phi)

    w /= np.max(np.abs(w))

    attrs = {'kind': 'berlage', 'frequency': np.squeeze(f, axis=1)}
    return _wrap_xarray(t, f, w, attrs)


def generalized(duration, dt, f, u=2, t=None, imag=False, sym=True):
    """
    Wang's generalized wavelet, of which the Ricker is a special case where
    u = 2. The parameter u is the order of the time-domain derivative, which
    can be a fractional derivative.

    As given by Wang (2015), Generalized seismic wavelets. GJI 203, p 1172-78.
    DOI: https://doi.org/10.1093/gji/ggv346. I am using the (more accurate)
    frequency domain method (eq 4 in that paper).

    .. plot::

        import matplotlib.pyplot as plt
        import bruges
        w, t = bruges.filters.generalized(0.256, 0.002, 40, u=1.0)
        plt.plot(t, w)

    Args:
        duration (float): The length of the wavelet, in s.
        dt (float): The time sample interval in s.
        f (float or array-like): The frequency or frequencies, in Hertz.
        u (float): The fractional derivative parameter u.
        t (array-like): The time series to evaluate at, if you don't want one
            to be computed. If you pass `t` then `duration` and `dt` will be
            ignored, so we recommend passing `None` for those arguments.
        center (bool): Whether to center the wavelet on time 0.
        imag (bool): Whether to return the imaginary component as well.
        sym (bool): If True (default behaviour before v0.5) then the wavelet
            is forced to have an odd number of samples and the central sample
            is at 0 time.

    Returns:
        xarray.DataArray. If f is a float, the resulting wavelet has
            duration/dt = A samples. If you give f as an array of length M,
            then the resulting wavelet bank will have shape (M, A).
    """
    # Make sure we can do banks.
    f = np.asanyarray(f).reshape(-1, 1)

    # Compute time domain response.
    if t is None:
        t = _get_time(duration, dt, sym=sym)
    else:
        if (duration is not None) or (dt is not None):
            m = "`duration` and `dt` are ignored when `t` is passed."
            warnings.warn(m, UserWarning, stacklevel=2)
        dt = t[1] - t[0]
        duration = len(t) * dt

    # Basics.
    om0 = f * 2 * np.pi
    u2 = u / 2
    df = 1 / duration
    nyquist = (1 / dt) / 2
    nf = 1 + nyquist / df
    t0 = duration / 2
    om = 2 * np.pi * np.arange(0, nyquist, df)

    # Compute the spectrum from Wang's eq 4.
    exp1 = np.exp((-om**2 / om0**2) + u2)
    exp2 = np.exp(-1j*om*t0 + 1j*np.pi * (1 + u2))
    W = (u2**(-u2)) * (om**u / om0**u) * exp1 * exp2

    w = np.fft.ifft(W, t.size)
    if not imag:
        w = w.real

    # At this point the wavelet bank has the shape (u, f, a),
    # where u is the size of u, f is the size of f, and a is
    # the number of amplitude samples we generated.
    w_max = np.max(np.abs(w), axis=-1)[:, None]
    w /= w_max

    attrs = {'kind': 'generalized', 'frequency': np.squeeze(f, axis=1)}
    return _wrap_xarray(t, f, w, attrs)
