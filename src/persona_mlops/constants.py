"""Shared dataset and feature names."""

DATASET_ID = "nvidia/Nemotron-Personas-Vietnam"
TEXT_COLUMN = "professional_persona"
TARGET_COLUMN = "region"

REQUIRED_COLUMNS = {
    "uuid", TEXT_COLUMN, TARGET_COLUMN, "country", "zone", "age", "sex",
    "marital_status", "education_level", "occupation", "sports_persona",
    "arts_persona", "travel_persona", "culinary_persona", "persona",
    "cultural_background", "skills_and_expertise", "hobbies_and_interests",
    "career_goals_and_ambitions",
}
