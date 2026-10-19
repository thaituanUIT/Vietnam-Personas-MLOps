from __future__ import annotations

import pandas as pd
import pytest

from persona_mlops.constants import REQUIRED_COLUMNS


@pytest.fixture
def persona_frame() -> pd.DataFrame:
    """Small deterministic fixture with all source columns and three regions."""
    rows = []
    regions = ["North", "Central", "South"]
    for index in range(60):
        region = regions[index % len(regions)]
        rows.append({
            "uuid": f"persona-{index:04d}",
            "professional_persona": (
                f"{region} region worker manages customer service and local business "
                f"with experience {index % 5}"
            ),
            "sports_persona": f"Enjoys walking and outdoor activity {index % 4}",
            "arts_persona": f"Enjoys local music and community events {index % 4}",
            "travel_persona": f"Visits regional cultural places {index % 4}",
            "culinary_persona": f"Prepares traditional meals {index % 4}",
            "persona": f"Synthetic adult persona number {index}",
            "cultural_background": f"Community background {region}",
            "skills_and_expertise": f"Communication and planning {index % 5}",
            "skills_and_expertise_list": ["communication", "planning"],
            "hobbies_and_interests": f"Reading and community activities {index % 4}",
            "hobbies_and_interests_list": ["reading", "community"],
            "career_goals_and_ambitions": f"Develop professional skills {index % 3}",
            "sex": "Nam" if index % 2 else "Nữ",
            "age": 20 + index % 50,
            "marital_status": "Độc thân" if index % 2 else "Đã kết hôn",
            "education_level": "Đại học" if index % 3 else "Trung học phổ thông",
            "occupation": f"Occupation {index % 6}",
            "zone": "Đô Thị" if index % 2 else "Nông Thôn",
            "region": region,
            "country": "Việt Nam",
        })
    frame = pd.DataFrame(rows)
    assert REQUIRED_COLUMNS.issubset(frame.columns)
    return frame
