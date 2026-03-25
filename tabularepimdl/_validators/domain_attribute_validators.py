"""Resuable validators built for attributes and domain constraint."""
from pydantic import model_validator

def domain_membership_validator(
    attribute_fields: tuple[str, ...],
    domain_fields: tuple[str, ...],
):
    """
    Ensure values from attribute_fields exist in the union of domain_fields.

    Returns:
        The model validator with a specific signature.

    Raises:
        ValueError: If an attribute field value does not exist in the domain field values.
    """
    @model_validator(mode="after")
    def validator(self):

        domain = set()
        for field in domain_fields:
            domain.update(getattr(self, field))

        for field in attribute_fields:
            value = getattr(self, field)

            if value not in domain:
                raise ValueError(
                    f"{field}='{value}' must exist in {domain_fields}. "
                    f"Allowed values: {sorted(domain)}."
                )

        return self

    return validator