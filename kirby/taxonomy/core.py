from enum import Enum


class StringIntEnum(Enum):
    """Enum where the value is a string, but can be cast to an int."""

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value

    @classmethod
    def from_string(cls, string: str) -> "StringIntEnum":
        r"""Convert a string to an enum member. This method is case insensitive and
        will replace spaces with underscores.

        Args:
             string: The string to convert to an enum member.

        Examples:
            >>> from kirby.taxonomy import Sex
            >>> Sex.from_string("Male")
            <Sex.MALE: 1>
            >>> Sex.from_string("M")
            <Sex.MALE: 1>
        """
        # normalize string by replacing spaces with underscores and converting
        # to upper case
        normalized_string = string.strip().upper().replace(" ", "_")
        # create a mapping of enum names to enum members
        mapping = {name.upper(): member for name, member in cls.__members__.items()}
        # try to match the string to an enum name
        if normalized_string in mapping:
            return mapping[normalized_string]
        # if there is no match raise an error
        raise ValueError(f"Could not find {string} in {cls.__name__}")
