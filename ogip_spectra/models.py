import astropy.units as u
import numpy as np
from gammapy.modeling.models import SpectralModel, Parameter, Parameters
from astromodels.functions.function import CompositeFunction


class XspecSpectralModel(SpectralModel):
    """A wrapper for XSPEC spectral models.

    Parameters
    ----------
    xspec_model : `~astromodels.functions.function.FunctionMeta`
        An instance of an XSPEC spectral model defined in ...
        CAVEAT: the energy units for the parameters are supposed to be in keV, and the flux units in keV-1 cm-2 s-1!
    """

    tag = ["XspecSpectralModel", "xspec"]

    def __init__(self, xspec_model):
        if isinstance(xspec_model, CompositeFunction):
            raise ValueError(
                "Composite functions are currently not supported. Please define each "
                "function as a separate XspecSpectralModel and compose them as normal"
                "SpectralModel instances."
            )
        self.xspec_model = xspec_model
        self._set_units()
        self.default_parameters = self._wrap_parameters()
        super().__init__()

    def _set_units(self):
        if self.xspec_model.has_fixed_units() is True:
            in_x_unit, in_y_unit = self.xspec_model.fixed_units
        else:
            in_x_unit = u.keV
            in_y_unit = "keV-1 cm-2 s-1"
        self.xspec_model.set_units(in_x_unit, in_y_unit)

    def _wrap_parameters(self):
        parameters = []
        for par in self.xspec_model._get_children():
            parameter = Parameter(
                name=par.name,
                value=par.value,
            )
            # TODO: set min, max, unit, ...
            parameters.append(parameter)
        return Parameters(parameters)

    def evaluate(self, energy, **kwargs):
        if not isinstance(energy, u.Quantity):
            raise ValueError("The energy must be a Quantity object.")
        else:
            energy = energy.to(self.xspec_model.x_unit)
        energy = np.array(energy) * energy.unit
        shape = energy.shape
        energy = energy.flatten()

        flux = self.xspec_model.evaluate(energy, **kwargs)
        return flux.reshape(shape)
