#this module may not be needed/used
#This module should know about a Pydantic model, 
# but it should not specifically know that the model is tabularepimdl.
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


from .yaml_exceptions import EpiModelDataError


ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_model(
    model_type: type[ModelT],
    data: dict[str, Any],
) -> ModelT:
    """
    Validate parsed YAML data against a Pydantic model.

    Args:
        model_type:
            Pydantic model class to validate against.

        data:
            Parsed YAML mapping.

    Returns:
        Validated Pydantic model instance.

    Raises:
        EpiModelDataError:
            If the YAML data does not satisfy the model schema.
    """

    try:
        return model_type.model_validate(data)

    except ValidationError as exc:
        raise EpiModelDataError(
            f"YAML data does not satisfy "
            f"{model_type.__name__} schema:\n{exc}"
        ) from exc
