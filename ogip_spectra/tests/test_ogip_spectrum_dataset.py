# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.maps import Map, RegionGeom, MapAxis
from ogip_spectra.ogip_spectrum_dataset import StandardOGIPDataset

@pytest.fixture()
def simple_geom():
    ener = np.linspace(0.1, 10, 50) * u.keV
    energy = MapAxis.from_edges(ener, name="energy", interp="lin")
    return RegionGeom.create(region=None, axes=[energy])


def test_create(simple_geom):
    counts = Map.from_geom(simple_geom)
    counts += 1
    counts_off = Map.from_geom(simple_geom)
    counts_off += 10
    grouping = Map.from_geom(simple_geom)
    grouping.data[...] = 1

    dataset = StandardOGIPDataset(
                        counts=counts,
                        counts_off=counts_off,
                        grouping=grouping
                        )

    assert dataset.grouping.geom.region == None
    assert_allclose(dataset.grouping.data, 1)
