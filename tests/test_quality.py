from pathlib import Path

import pytest

from persona_mlops.quality import validate


def test_validate_writes_quality_report(tmp_path: Path, persona_frame):
    report_path = tmp_path / "reports" / "quality.json"

    report = validate(persona_frame, report_path)

    assert report["rows"] == 60
    assert report["duplicate_uuid_count"] == 0
    assert report["empty_text_count"] == 0
    assert report_path.is_file()


def test_validate_rejects_missing_schema_column(persona_frame, tmp_path: Path):
    frame = persona_frame.drop(columns=["occupation"])

    with pytest.raises(ValueError, match="occupation"):
        validate(frame, tmp_path / "quality.json")


def test_validate_rejects_duplicate_ids(persona_frame, tmp_path: Path):
    persona_frame.loc[1, "uuid"] = persona_frame.loc[0, "uuid"]

    with pytest.raises(ValueError, match="duplicate_uuid_count"):
        validate(persona_frame, tmp_path / "quality.json")


def test_validate_rejects_null_contract_fields(persona_frame, tmp_path: Path):
    persona_frame.loc[0, "region"] = None

    with pytest.raises(ValueError, match="Data quality checks failed"):
        validate(persona_frame, tmp_path / "quality.json")
