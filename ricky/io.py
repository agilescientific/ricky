import pandas as pd
import xarray as xr


def read_csv(fname, time='time', amplitude='amplitude', offset='offset', **kwargs):
    """
    Read from a CSV file.
    """
    df = pd.read_csv(fname, **kwargs)

    t_ = xr.DataArray(t, dims=['time'], attrs={'units': 's'})
    w_ = xr.DataArray(w, name='amplitude', coords={'time': t_})

    return 