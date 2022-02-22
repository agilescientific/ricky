import xarray as xr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

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
