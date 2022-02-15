# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

from gammapy.maps import Map, RegionGeom, MapAxis
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.modeling import Fit
from gammapy.datasets import Datasets

from ..ogip_spectrum_dataset import StandardOGIPDataset


@pytest.fixture()
def simple_geom():
    ener = np.linspace(0.1, 10, 51) * u.keV
    energy = MapAxis.from_edges(ener, name="energy", interp="lin")
    return RegionGeom.create(region=None, axes=[energy])


@pytest.fixture()
def ogip_dataset():
    return StandardOGIPDataset.read(
        filename="/home/lucagiunti/gammapy-ogip-spectra/XMM_test_files/PN_PWN.grp"
    )


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
        grouping_axis=grouping_axis,
    )

    assert dataset.grouping_axis.nbin == 25


def test_read(ogip_dataset):
    assert ogip_dataset.counts.data.sum() == 789
    assert ogip_dataset.counts_off.data.sum() == 421
    print(ogip_dataset.grouped.counts.data)
    assert np.all(ogip_dataset.grouped.counts.data[:38] >= 20)


def test_fit(ogip_dataset):
    spectral_model = PowerLawSpectralModel(
        reference="1 keV", amplitude="1e-3 cm-2s-1 keV-1"
    )
    model = SkyModel(spectral_model=spectral_model)

    # To test the mask_fit handling, but also to avoid low-energy bins where absorption is relevant
    ogip_dataset.mask_fit = ogip_dataset._geom.energy_mask(5 * u.keV, 10 * u.keV)

    datasets = Datasets([ogip_dataset])
    datasets.models = [model]
    assert datasets.models[0] == ogip_dataset.grouped.models[0]

    fit = Fit()
    fit_result = fit.run(datasets)

    assert fit_result.success is True
    assert_allclose(fit_result.total_stat, -794.9787492685413)
    parameters = fit_result.parameters
    assert_allclose(parameters["amplitude"].value, 0.003992296953446386)
    assert_allclose(parameters["index"].error, 0.822836108835805)
