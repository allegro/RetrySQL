from enum import Enum


class ListConvertableEnum(Enum):
    @classmethod
    def get_values(cls) -> list[str]:
        return list(map(lambda field: field.value, cls))
