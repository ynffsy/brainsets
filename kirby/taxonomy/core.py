from enum import Enum

class StringIntEnum(Enum):
    """Enum where the value is a string, but can be cast to an int."""

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value
