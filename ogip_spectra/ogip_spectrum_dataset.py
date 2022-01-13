# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.datasets import SpectrumDatasetOnOff

class StandardOGIPDataset(SpectrumDatasetOnOff):
    """Dataset containing spectral data as defined by X-ray OGIP compliant files.

    A few elements are added that are not supported by the current SpectrumDataset in gammapy.

    grouping contains the information on the grouping scheme, namely the group number of each bin.

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
    grouping_axis : `~gammapy.maps.MapAxis`
        MapAxis defining the grouping scheme.
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
        axis = kwargs.pop("grouping_axis",None)
        super().__init__(*args, **kwargs)
        self._grouped_dataset = None
        self.grouping_axis = axis

    @property
    def grouping_axis(self):
        """Energy axis providing energy grouping for stat calculations."""
        return self._grouping_axis

    @grouping_axis.setter
    def grouping_axis(self, axis):
        """Energy axis providing energy grouping for stat calculations."""
        self._grouping_axis = axis
        if axis is not None:
            self._apply_grouping()

    def _apply_grouping(self):
        """Apply grouping."""
        self._grouped_dataset = self.resample_energy_axis(self.grouping_axis, name=f"group_{self.name}")

    @property
    def grouped(self):
        """Return grouped dataset."""
        return self._grouped_dataset

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        return self.grouped.stat_sum()

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder.

        For formats specs see `OGIPDatasetReader.read`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read
        """
        from .io import StandardOGIPDatasetReader

        reader = StandardOGIPDatasetReader(filename=filename)
        return reader.read()

    def write(self, filename, overwrite=False, format="ogip"):
        raise NotImplementedError("Standard OGIP writing is not supported.")

