"""Deterministic, invented Vietnamese records for an offline ETL demonstration."""

import pandas as pd

from persona_mlops.constants import REQUIRED_COLUMNS


def demo_frame() -> pd.DataFrame:
    rows = []
    regions = ["Đồng bằng sông Hồng", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"]
    jobs = ["Giáo viên", "Nhân viên khách sạn", "Kỹ sư"]
    for index in range(120):
        occupation = jobs[index % 3]
        row = {column: "Mô tả minh họa về sinh hoạt và sở thích cá nhân." for column in REQUIRED_COLUMNS}
        row.update(
            uuid=f"demo-{index:04d}", age=18 + index % 60,
            sex="Nam" if index % 2 else "Nữ", country="Việt Nam",
            region=regions[index % 3], zone="Đô Thị" if index % 2 else "Nông Thôn",
            occupation=occupation, marital_status="Độc thân",
            education_level="Đại học",
            professional_persona=f"Làm công việc {occupation}, có kỹ năng giao tiếp và quản lý thời gian.",
        )
        rows.append(row)
    return pd.DataFrame(rows)
