"""
xarray accessor.

Author: Matt Hall
Email: matt@agilescientific.com
Licence: Apache 2.0
"""
import xarray as xr
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

from . import utils


@xr.register_dataarray_accessor("ricky")
class WaveletAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def wiggle(self,
               perc=99,
               gain=1,
               skip=1,
               ax=None,
               orientation='vertical',
               rgb=(0,0,0),
               alpha=1,
               lw=1,
               oversampling=1,
              ):
        """
        Args:
            perc (float): Percentile to scale to, default 99%.
            gain (float): Gain value.
            skip (int): Skip=1, every trace, skip=2, every second trace, etc.
            ax (Axes): matplotlib Axes object (optional).
            orientation (str): 'horizontal' or 'vertical'.
            rgb (tuple): 3-tuple of RGB for the trace colour.
            alpha (float): Opacity. Default 1.0.
            lw (float): Lineweight. Default 0.5.
            oversampling (int): The approximate number of new samples per sample.
                Higher numbers result in smoother interpolated wavelets.
        Returns:
            Axes: A matplotlib Axes object.
        """
        ntraces, nt = self._obj.shape
        if ax is None:
            space = min(10, 1.0+ntraces/(skip*2))
            if orientation.casefold().startswith('v'):
                figsize = (space, 4)
            else:
                figsize = (6, space)
            fig, ax = plt.subplots(figsize=figsize)
        rgba = list(rgb) + [alpha]
        sc = np.percentile(self._obj, perc)  # Normalization factor
        t = self._obj.time
        hypertime = np.linspace(t[0], t[-1], (oversampling * t.size - 1) + 1)
        wigdata = self._obj[::skip, :]
        xpos = self._obj.coords['frequency'][::skip]

        for x, trace in zip(xpos.data, wigdata):
            # Compute high resolution trace.
            amp = gain * trace / sc + x
            interp = interp1d(t, amp, kind='cubic')
            hyperamp = interp(hypertime)

            # Plot the line.
            if orientation.casefold().startswith('v'):
                ax.plot(hyperamp, hypertime, 'k', lw=lw)
                fill = ax.fill_betweenx
                ax.set_ylabel('time [s]')
                if ntraces > 2:
                    ax.set_xlabel('frequency [Hz]')
                    ax.set_title(f'{self._obj.kind}')
                else:
                    ax.set_xlabel('amplitude')
                    ax.set_title(f'{self._obj.kind}\nf = {x} [Hz]')
            elif orientation.casefold().startswith('h'):
                ax.plot(hypertime, hyperamp, 'k', lw=lw)
                fill = ax.fill_between
                ax.set_xlabel('time [s]')
                if ntraces > 2:
                    ax.set_ylabel('frequency [Hz]')
                    ax.set_title(self._obj.kind)
                else:
                    ax.set_ylabel('amplitude')
                    ax.set_title(f'{self._obj.kind}\nfrequency = {x} [Hz]')
            else:
                raise ValueError(f"`orientation` must be 'vertical' or 'horizontal'")

            # Plot the fill.
            fill(hypertime, hyperamp, x,
                where=hyperamp > x,
                facecolor=rgba,
                lw=0, interpolate=True,
                )

        return ax

    def rotate_phase(self, phi, degrees=False):
        r"""
        Performs a phase rotation of wavelet or wavelet bank using:

        .. math::

            A = w(t)\cos\phi - h(t)\sin\phi

        where `w(t)` is the wavelet, `h(t)` is its Hilbert transform, and \phi is
        the phase rotation angle (default is radians).

        The analytic signal can be written in the form :math:`S(t) = A(t)e^{j\theta (t)}`
        where :math:`A(t) = \left| h(w(t)) \right|` and :math:`\theta(t) = \tan^{-1}[h(w(t))]`. 
        `A(t)` is called the "reflection strength" and :math:`\phi(t)` is called the "instantaneous
        phase".

        A constant phase rotation :math:`\phi` would produce the analytic signal
        :math:`S(t)=A(t)e^{j(\theta(t) + \phi)}`. To get the non-analytic signal,
        we take 

        .. math::

            real(S(t)) &= A(t)\cos(\theta(t) + \phi) \\
            &= A(t)\cos\theta(t)\cos(\phi)-\sin\theta(t)\sin(\phi))\\
            &= w(t)\cos\phi-h(t)\sin\phi
            

        Args:
            w (ndarray): The wavelet vector, can be a 2D wavelet bank.
            phi (float): The phase rotation angle (in radians) to apply.
            degrees (bool): If phi is in degrees not radians.

        Returns:
            The phase rotated signal (or bank of signals).
        """
        # Make sure the data is at least 2D to apply_along
        data = np.atleast_2d(self._obj.values)

        # Get Hilbert transform. This will be 2D.
        a = utils.apply_along_axis(scipy.signal.hilbert, data, axis=0)

        # Transform angles into what we need.
        phi = np.asanyarray(phi).reshape(-1, 1, 1)
        if degrees:
            phi = np.radians(phi)

        rotated = np.real(a) * np.cos(phi)  -  np.imag(a) * np.sin(phi)
        return np.squeeze(rotated)


# TODO:
# - Add a to_segy method.
# - Add phase rotation?
# - Add spectrum.
# - to_csv, to_json etc are easier via xarray or pandas.
