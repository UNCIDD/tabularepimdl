"""Resuable validators built for attributes and domain constraint."""
from collections.abc import Iterable
from pydantic import model_validator


def normalize_fields(name, fields):
    if isinstance(fields, str):
            raise TypeError(
                f"{name} of domain_membership_validator must be a tuple/list of field names, not a string. "
                f"Did you mean ({fields!r},)?"
            )
    
    if not isinstance(fields, Iterable):
            raise TypeError(f"{name} must be an iterable of strings.")
    
    fields = tuple(fields)

    return fields

def domain_membership_validator(
    attribute_fields: Iterable[str],
    domain_fields: Iterable[str],
):
    """
    Ensure values from attribute_fields exist in the union of domain_fields.

    Returns:
        The model validator with a specific signature.

    Raises:
        ValueError: If an attribute field value does not exist in the domain field values.
    """
    attribute_fields_ = normalize_fields("attribute_fields", attribute_fields)
    domain_fields_ = normalize_fields("domain_fields", domain_fields)

    @model_validator(mode="after")
    def validator(self):
        #build domain set
        domain = set()
        for field in domain_fields_:
            if not hasattr(self, field):
                 raise ValueError(f"Rule has no attribute '{field}'.")
            
            domain_values = getattr(self, field, None)

            if domain_values is None:
                continue
            
            if isinstance(domain_values, str) or not isinstance(domain_values, Iterable):
                domain.add(domain_values)
            else:
                domain.update(domain_values)
        

        #validate attribute fields
        for field in attribute_fields_:
            if not hasattr(self, field):
                 raise ValueError(f"Rule has no attribute '{field}'.")
            
            attribute_value = getattr(self, field, None)
            

            if attribute_value is None:
                continue

            # Normalize to iterable
            if isinstance(attribute_value, str) or not isinstance(attribute_value, Iterable):
                values_to_check = [attribute_value]
            elif isinstance(attribute_value, Iterable):
                values_to_check = attribute_value
            
            invalid_values = [v for v in values_to_check if v not in domain]

            if invalid_values:
                raise ValueError(
                    f"{field} contains invalid value(s): {invalid_values}. "
                    f"Allowed values are: {sorted(domain)}."
                )
            
        return self

    return validator