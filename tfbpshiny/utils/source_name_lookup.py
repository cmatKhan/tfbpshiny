import logging
from enum import Enum
from typing import Literal

logger = logging.getLogger("shiny")


class BindingSource(str, Enum):
    harbison_chip = "ChIP-chip"
    chipexo_pugh_allevents = "ChIP-exo"
    brent_nf_cc = "Calling Cards"


class PerturbationSource(str, Enum):
    mcisaac_oe = "Overexpression"
    kemmeren_tfko = "2014 TFKO"
    hu_reimann_tfko = "2007 TFKO"
    hahn_degron = "Degron"


def get_source_name_dict(
    datatype: Literal["binding", "perturbation_response"] | None = None,
    reverse: bool = False,
) -> dict[str, str]:
    """
    Return a mapping between internal source names and display labels for a given
    datatype.

    This function provides a consistent dictionary mapping for source names,
    derived from enum classes. It supports filtering by datatype and reversing the
    direction of the mapping.

    :param datatype: Optional; one of "binding", "perturbation_response", or None.
        If provided, limits the dictionary to that specific datatype.
        If None (default), includes both types.
    :type datatype: Optional[Literal["binding", "perturbation_response"]]
    :param reverse: If True, reverses the mapping so that display names map to
        internal names.
    :type reverse: bool
    :return: A dictionary mapping internal source names to display names,
        or the reverse.
    :rtype: dict[str, str]

    :raises ValueError: If an invalid datatype is provided.

    :example:

        >>> get_source_name_dict()
        {'harbison_chip': 'ChIP-chip', 'mcisaac_oe': 'Overexpression', ...}

        >>> get_source_name_dict(datatype="binding")
        {'harbison_chip': 'ChIP-chip', 'chipexo_pugh_allevents': 'ChIP-exo', ...}

        >>> get_source_name_dict(reverse=True)
        {'ChIP-chip': 'harbison_chip', 'Overexpression': 'mcisaac_oe', ...}

        >>> get_source_name_dict(datatype="perturbation_response", reverse=True)
        {'Overexpression': 'mcisaac_oe', '2014 TFKO': 'kemmeren_tfko', ...}

    """
    enum_class: type[Enum]
    if datatype == "binding":
        enum_class = BindingSource
    elif datatype == "perturbation_response":
        enum_class = PerturbationSource
    elif datatype is None:
        mapping = {
            **{k.name: k.value for k in BindingSource},
            **{k.name: k.value for k in PerturbationSource},
        }
        return {v: k for k, v in mapping.items()} if reverse else mapping
    else:
        raise ValueError(f"Invalid datatype: {datatype}")

    mapping = {k.name: k.value for k in enum_class}
    return {v: k for k, v in mapping.items()} if reverse else mapping
