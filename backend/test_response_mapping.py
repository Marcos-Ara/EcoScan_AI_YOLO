"""Offline test for the Roboflow -> EcoScan response mapping."""

import os
import sys

# This script only tests normalization and does not make a network request.
os.environ.setdefault("ROBOFLOW_API_KEY", "test-only")

from app import normalize_roboflow_result

CASES = [
    ("Metal", "Metal", "metal"),
    ("Cardboard", "Papel", "papel"),
    ("Paper", "Papel", "papel"),
    ("Plastic", "Plástico", "plastico"),
    ("Glass", "Vidro", "vidro"),
]

for source_class, expected_category, expected_key in CASES:
    result, _ = normalize_roboflow_result(
        {
            "predictions": [
                {
                    "class": source_class,
                    "confidence": 0.91,
                    "x": 50,
                    "y": 50,
                    "width": 40,
                    "height": 60,
                }
            ]
        },
        0,
        0,
        100,
        100,
    )

    assert result, f"Sem resultado para {source_class}"
    assert result[0]["category"] == expected_category
    assert result[0]["category_key"] == expected_key

print("Roboflow response mapping: OK")
