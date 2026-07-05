from enum import Enum

import pytest

from nanny.activity import ActivityError, BabyActivity


def test_validate_accepts_well_formed_record():
    a = BabyActivity(
        timestamp="2026-07-05T15:00:00+00:00",
        activity_type="bottle",
        quantity=4.0,
        unit="oz",
        notes="",
    )
    assert a.validate() is a


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp", ""),
        ("timestamp", "not-a-date"),
        ("activity_type", "nap"),
        ("unit", "liters"),
        ("quantity", -1.0),
    ],
)
def test_validate_rejects_bad_field(field, value):
    kwargs = {
        "timestamp": "2026-07-05T15:00:00+00:00",
        "activity_type": "bottle",
        "quantity": 4.0,
        "unit": "oz",
        "notes": "",
    }
    kwargs[field] = value
    with pytest.raises(ActivityError):
        BabyActivity(**kwargs).validate()


def test_from_dict_ignores_unknown_keys_and_coerces_quantity():
    a = BabyActivity.from_dict(
        {
            "timestamp": "2026-07-05T15:00:00+00:00",
            "activity_type": "solids",
            "quantity": "50",
            "unit": "grams",
            "notes": "",
            "unexpected_field": "ignored",
        }
    )
    assert a.quantity == 50.0
    assert not hasattr(a, "unexpected_field")


def test_from_dict_coerces_enum_like_values_to_plain_str():
    # Regression test: a structured-output schema field backed by an Enum
    # (e.g. a classic `Enum(..., type=str)` mixin) round-trips through
    # pydantic's model_dump() as the raw enum member, whose str() is
    # "ClassName.member" rather than the plain value. from_dict must coerce
    # it to a plain string so f-string interpolation never leaks that repr.
    LeakyEnum = Enum("LeakyEnum", {"poop": "poop"}, type=str)
    a = BabyActivity.from_dict(
        {
            "timestamp": "2026-07-05T15:00:00+00:00",
            "activity_type": LeakyEnum.poop,
            "quantity": 1.0,
            "unit": "count",
            "notes": "",
        }
    )
    assert a.activity_type == "poop"
    assert str(a.activity_type) == "poop"
    assert isinstance(a.activity_type, str)
    assert type(a.activity_type) is str
