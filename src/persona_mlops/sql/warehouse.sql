-- Full-snapshot load. Content-derived keys are stable across reruns and row order.
-- The Python caller executes this entire model in one transaction.
DROP TABLE IF EXISTS gold.fact_model_predictions;
DROP VIEW IF EXISTS gold.mart_audience_summary;
DROP TABLE IF EXISTS gold.fact_persona;
DROP TABLE IF EXISTS gold.dim_geography;
DROP TABLE IF EXISTS gold.dim_demographics;
DROP TABLE IF EXISTS gold.dim_occupation;

CREATE TABLE gold.dim_geography (
    geography_key VARCHAR PRIMARY KEY, country VARCHAR NOT NULL,
    region VARCHAR NOT NULL, zone VARCHAR NOT NULL
);
INSERT INTO gold.dim_geography
SELECT md5(to_json(struct_pack(country := country, region := region, zone := zone))) AS geography_key,
       country, region, zone
FROM (SELECT DISTINCT country, region, zone FROM silver.personas);

CREATE TABLE gold.dim_demographics (
    demographics_key VARCHAR PRIMARY KEY, sex VARCHAR NOT NULL, age INTEGER NOT NULL,
    marital_status VARCHAR NOT NULL, education_level VARCHAR NOT NULL
);
INSERT INTO gold.dim_demographics
SELECT md5(to_json(struct_pack(sex := sex, age := age, marital_status := marital_status,
                              education_level := education_level))) AS demographics_key,
       sex, age, marital_status, education_level
FROM (SELECT DISTINCT sex, age, marital_status, education_level FROM silver.personas);

CREATE TABLE gold.dim_occupation (
    occupation_key VARCHAR PRIMARY KEY, occupation VARCHAR NOT NULL
);
INSERT INTO gold.dim_occupation
SELECT md5(to_json(struct_pack(occupation := occupation))) AS occupation_key, occupation
FROM (SELECT DISTINCT occupation FROM silver.personas);

-- Factless snapshot fact: one row per synthetic persona UUID, not an event.
CREATE TABLE gold.fact_persona (
    persona_id VARCHAR PRIMARY KEY,
    geography_key VARCHAR NOT NULL REFERENCES gold.dim_geography(geography_key),
    demographics_key VARCHAR NOT NULL REFERENCES gold.dim_demographics(demographics_key),
    occupation_key VARCHAR NOT NULL REFERENCES gold.dim_occupation(occupation_key),
    professional_persona VARCHAR, sports_persona VARCHAR, arts_persona VARCHAR,
    travel_persona VARCHAR, culinary_persona VARCHAR, persona VARCHAR,
    cultural_background VARCHAR, skills_and_expertise VARCHAR,
    hobbies_and_interests VARCHAR, career_goals_and_ambitions VARCHAR
);
INSERT INTO gold.fact_persona
SELECT p.uuid, g.geography_key, d.demographics_key, o.occupation_key,
       p.professional_persona, p.sports_persona, p.arts_persona, p.travel_persona,
       p.culinary_persona, p.persona, p.cultural_background, p.skills_and_expertise,
       p.hobbies_and_interests, p.career_goals_and_ambitions
FROM silver.personas p
JOIN gold.dim_geography g USING (country, region, zone)
JOIN gold.dim_demographics d USING (sex, age, marital_status, education_level)
JOIN gold.dim_occupation o USING (occupation);

CREATE VIEW gold.mart_audience_summary AS
SELECT g.region, g.zone, o.occupation, count(*) AS persona_count,
       round(avg(d.age), 2) AS average_age
FROM gold.fact_persona f
JOIN gold.dim_geography g USING (geography_key)
JOIN gold.dim_demographics d USING (demographics_key)
JOIN gold.dim_occupation o USING (occupation_key)
GROUP BY g.region, g.zone, o.occupation;
