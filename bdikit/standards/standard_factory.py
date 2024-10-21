from enum import Enum
from typing import Mapping, Any, Type
from bdikit.standards.base import BaseStandard
from bdikit.standards.gdc import GDC


class Standards(Enum):
    GDC = ("gdc", GDC)

    def __init__(self, standard_name: str, standard_class: Type[BaseStandard]):
        self.standard_name = standard_name
        self.standard_class = standard_class

    @staticmethod
    def get_instance(
        standard_name: str, **standard_kwargs: Mapping[str, Any]
    ) -> BaseStandard:
        standards = {
            standard.standard_name: standard.standard_class for standard in Standards
        }
        try:
            return standards[standard_name](**standard_kwargs)
        except KeyError:
            names = ", ".join(list(standards.keys()))
            raise ValueError(
                f"The {standard_name} standard is not supported. "
                f"Supported standards are: {names}"
            )
