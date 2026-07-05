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
