import logging

from .Rule import Rule

# The pandas-structured rules/engine (BirthProcess, EpiModel, etc.) that used to be exported here
# have moved to legacy/pandas_reference/. They were kept only as an internal reference baseline 
# to validate the NumPy-structured rules/engine
# See legacy/README.md.

### The following rules are in vectorized/numpy structure ###
from .BirthProcess_Vec_Encode import BirthProcess_Vec_Encode
from .EnvironmentalTransmission_Vec_Encode import EnvironmentalTransmission_Vec_Encode
from .HospRule_Vec_Encode import HospRule_Vec_Encode
from .MultiStrainInfectiousProcess_Vec_Encode import MultiStrainInfectiousProcess_Vec_Encode
from .SharedTraitInfection_Vec_Encode import SharedTraitInfection_Vec_Encode
from .SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from .SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode
from .SimpleObservationProcess_Vec_Encode import SimpleObservationProcess_Vec_Encode
from .StateBasedDeathProcess_Vec_Encode import StateBasedDeathProcess_Vec_Encode
from .WAIFWTransmission_Vec_Encode_Bincount import WAIFWTransmission_Vec_Encode_Bincount

from .EpiModel_Vec_Encode1_5 import EpiModel_Vec_Encode_1_5

__version__ = "0.2.0"


def configure_logging(level: int = logging.DEBUG) -> None:
    """Turn on tabularepimdl's debug logging output.

    By default this package emits no log output at all -- rules and the model engine log
    intermediate-calculation details (state encodings, domain maps, etc.) at DEBUG level, but a
    library only emits log records. It's up to the application to decide whether to display them.
    Call this once (e.g. at the top of a script or notebook) to see that output while developing
    or troubleshooting a model.

    Args:
        level: the logging level to enable for tabularepimdl's logger hierarchy. Defaults to DEBUG.
    
    Usage Example:
        to use the package-level logging, add the import and function before other imports: 
        import tabularepimdl
        tabularepimdl.configure_logging()
        from tabularepimdl.<rule_name> import <class_name>
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    package_logger = logging.getLogger(__name__)
    package_logger.addHandler(handler)
    package_logger.setLevel(level)