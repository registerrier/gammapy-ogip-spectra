import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

from astromodels.xspec.factory import XS_wabs
from astromodels import Powerlaw

from ..models import XspecSpectralModel


def test_XspecSpectralModel():
    plaw = Powerlaw(index=-2, K=1e-3, piv=1)
    abs_model = XS_wabs(nh=5)
    energy_grid = np.linspace(0.5, 10.0, 1000) * u.keV

    # Gammapy wrapper
    f1 = XspecSpectralModel(plaw)
    f2 = XspecSpectralModel(abs_model)
    f3 = f1 * f2

    # Plain astromodels
    plaw_with_abs = plaw * abs_model

    assert_allclose(f3(energy_grid).value, plaw_with_abs(energy_grid.value))
    SkyModel(spectral_model=f3)  # Test evaluate on simple geom
    with pytest.raises(ValueError):
        SkyModel(spectral_model=f2)  # Wrong units, f2 is an absorption model
