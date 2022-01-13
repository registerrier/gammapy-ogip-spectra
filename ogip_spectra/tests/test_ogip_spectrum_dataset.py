# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.maps import Map, RegionGeom, MapAxis
from ogip_spectra.ogip_spectrum_dataset import StandardOGIPDataset

@pytest.fixture()
def simple_geom():
    ener = np.linspace(0.1, 10, 51) * u.keV
    energy = MapAxis.from_edges(ener, name="energy", interp="lin")
    return RegionGeom.create(region=None, axes=[energy])


def test_create(simple_geom):
    counts = Map.from_geom(simple_geom)
    counts += 1
    acceptance = Map.from_geom(simple_geom)
    acceptance += 1
    counts_off = Map.from_geom(simple_geom)
    counts_off += 10
    acceptance_off = Map.from_geom(simple_geom)
    acceptance_off += 1

    grouping_axis = simple_geom.axes[0].downsample(2)

    dataset = StandardOGIPDataset(
                        counts=counts,
                        counts_off=counts_off,
                        acceptance=acceptance,
                        acceptance_off=acceptance_off,
                        grouping_axis=grouping_axis
                        )

    assert dataset.grouping_axis.nbin == 25

def test_read():
    dataset = StandardOGIPDataset.read(filename="$OGIP_DATA/xmm/PN_PWN.grp")

    assert dataset.counts.data.sum() == 3316
    assert dataset.counts_off.data.sum() == 2879
    assert np.all(dataset.grouped.counts.data[:30]>=25)
