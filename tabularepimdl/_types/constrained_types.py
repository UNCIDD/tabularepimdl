from typing import Annotated
from pydantic import AfterValidator


def unique_nonempty_str_list(value: list[str]) -> list[str]:
    """Ensure the input list is provided and not empty. Ensure list elements are unique. Accept strings only."""
    if not value:
        raise ValueError("List must contain at least one element. Please provide all required elements.")
    if len(value) != len(set(value)):
        raise ValueError("List elements must be unique.")
    return value

UniqueNonEmptyStrList = Annotated[list[str], AfterValidator(unique_nonempty_str_list)]


def unique_nonempty_str_int_list(value: list[str | int]) -> list[str | int]:
    """Ensure the input list is provided and not empty. Ensure list elements are unique. Accept strings and intergers."""
    if not value:
        raise ValueError("List must contain at least one element. Please provide all required elements.")
    if len(value) != len(set(value)):
        raise ValueError("List elements must be unique.")
    return value

UniqueNonEmptyStrIntList = Annotated[list[str], AfterValidator(unique_nonempty_str_int_list)]