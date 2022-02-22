"""Test ricky"""
import pytest
import ricky


def test_ricker():
    """Test the basics.
    """
    # Test the basics.
    w = ricky.ricker(duration=0.1, dt=0.001, f=20)
    assert w.shape == (1, 101)

    w = ricky.ricker(duration=0.1, dt=0.001, f=[5, 10, 15, 20])
    assert w.shape == (4, 101)
    assert w.time.size == 101
    assert w.time.attrs['units'] == 's'
    assert w.frequency.size == 4
    assert w.frequency.attrs['units'] == 'Hz'

    with pytest.warns(UserWarning):
        # Raises warning because duration and dt are not used if t is passed.
        w = ricky.ricker(duration=0.1, dt=0.001, f=20, t=[-0.025, 0, 0.025])

    w = ricky.ricker(duration=None, dt=None, f=20, t=[-0.025, 0, 0.025])
    assert w.shape == (1, 3)


def test_sinc():
    """Test the sinc wavelet.
    """
    # Test the basics.
    w = ricky.sinc(duration=0.1, dt=0.001, f=20)
    assert w.shape == (1, 101)

    w = ricky.sinc(duration=0.1, dt=0.001, f=[5, 10, 15, 20])
    assert w.shape == (4, 101)
    assert w.time.size == 101
    assert w.time.attrs['units'] == 's'
    assert w.frequency.size == 4
    assert w.frequency.attrs['units'] == 'Hz'

    with pytest.warns(UserWarning):
        # Raises warning because duration and dt are not used if t is passed.
        w = ricky.sinc(duration=0.1, dt=0.001, f=20, t=[-0.025, 0, 0.025])

    # Should pass taper=None if arbitrary times are passed.
    w = ricky.sinc(duration=None, dt=None, f=20, t=[-0.025, 0, 0.025], taper=None)
    assert w.shape == (1, 3)
