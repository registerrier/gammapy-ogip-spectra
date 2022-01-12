# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.datasets import SpectrumDatasetOnOff

class StandardOGIPDataset(SpectrumDatasetOnOff):
    """Dataset containing spectral data as defined by X-ray OGIP compliant files.

    A few elements are added that are not supported by the current SpectrumDataset in gammapy.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    counts_off : `~gammapy.maps.WcsNDMap`
        Ring-convolved counts cube
    acceptance : `~gammapy.maps.WcsNDMap`
        Acceptance from the IRFs
    acceptance_off : `~gammapy.maps.WcsNDMap`
        Acceptance off
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel. Unused here.
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask_safe : `~gammapy.maps.WcsNDMap`
        Mask defining the safe data range.
    grouping : `~gammapy.maps.WcsNDMap`
        Map defining the grouping scheme.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    name : str
        Name of the dataset.

    """
    stat_type = "wstat"
    tag = "StandardOGIPDataset"

    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.grouping = kwargs.pop("grouping",None)
        super().__init__(*args, **kwargs)
