-- =====================================================================
-- 02_clean.sql : raw -> typed -> validated -> one row per policy with claims
-- Dataset: freMTPL2 (French motor third-party liability), freq + sev tables.
-- Target : average claim severity per policy = total (capped) claim cost / claim count.
-- =====================================================================

-- 1. Type & trim. OpenML ships string columns with embedded quotes ('B12'), so strip them.
DROP TABLE IF EXISTS stg_freq;
CREATE TABLE stg_freq AS
SELECT CAST(IDpol AS INTEGER)       AS idpol,
       CAST(ClaimNb AS INTEGER)     AS claim_nb,
       CAST(Exposure AS REAL)       AS exposure,
       TRIM(Area,     '''" ')       AS area,
       CAST(VehPower AS INTEGER)    AS veh_power,
       CAST(VehAge AS INTEGER)      AS veh_age,
       CAST(DrivAge AS INTEGER)     AS driv_age,
       CAST(BonusMalus AS INTEGER)  AS bonus_malus,
       TRIM(VehBrand, '''" ')       AS veh_brand,
       TRIM(VehGas,   '''" ')       AS veh_gas,
       CAST(Density AS REAL)        AS density,
       TRIM(Region,   '''" ')       AS region
FROM raw_freq;

-- 2. Validate + de-duplicate policies (keep the row with the largest exposure per idpol).
--    Rules are documented in README; rejected-row counts are reported in 03_checks.sql.
DROP TABLE IF EXISTS clean_freq;
CREATE TABLE clean_freq AS
WITH valid AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY idpol ORDER BY exposure DESC) AS rn
    FROM stg_freq
    WHERE idpol IS NOT NULL
      AND exposure > 0
      AND driv_age BETWEEN 18 AND 120
      AND veh_age >= 0
      AND bonus_malus >= 50
      AND density > 0
      AND area IS NOT NULL AND veh_brand IS NOT NULL
      AND veh_gas IS NOT NULL AND region IS NOT NULL
)
SELECT idpol, claim_nb,
       MIN(exposure, 1.0) AS exposure,          -- policies cannot be exposed > 1 year in this data
       area, veh_power, veh_age, driv_age, bonus_malus,
       veh_brand, veh_gas, density, region
FROM valid
WHERE rn = 1;
CREATE UNIQUE INDEX ix_clean_freq ON clean_freq(idpol);

-- 3. Claim-level severities: drop non-positive / null amounts.
DROP TABLE IF EXISTS clean_sev;
CREATE TABLE clean_sev AS
SELECT CAST(IDpol AS INTEGER) AS idpol, CAST(ClaimAmount AS REAL) AS claim_amount
FROM raw_sev
WHERE ClaimAmount IS NOT NULL AND CAST(ClaimAmount AS REAL) > 0;
CREATE INDEX ix_clean_sev ON clean_sev(idpol);

-- 4. Policy-level claim aggregation. Individual claims are capped at 100,000 (a standard
--    large-loss threshold; the freMTPL2 tail reaches ~4m from a handful of claims, which would
--    dominate any Gamma fit). n_claims is taken from the severity table itself, so numerator
--    and denominator are consistent even where freq.ClaimNb disagrees with it.
DROP TABLE IF EXISTS policy_sev;
CREATE TABLE policy_sev AS
SELECT idpol,
       COUNT(*)                                 AS n_claims,
       SUM(MIN(claim_amount, 100000.0))         AS total_capped,
       SUM(claim_amount)                        AS total_raw,
       MAX(claim_amount)                        AS max_claim
FROM clean_sev
GROUP BY idpol;

-- 5. Modelling table: policies with >= 1 claim + capped/banded features + deterministic split.
--    Split is a hash of the policy id (idpol % 5) so it is reproducible in SQL, R and Python.
DROP TABLE IF EXISTS model_severity;
CREATE TABLE model_severity AS
SELECT f.idpol,
       s.n_claims,
       s.total_capped / s.n_claims AS avg_severity,
       f.exposure,
       f.area,
       MIN(f.veh_power, 12)        AS veh_power_g,
       f.veh_power,
       MIN(f.veh_age, 20)          AS veh_age_c,
       MIN(f.driv_age, 90)         AS driv_age_c,
       MIN(f.bonus_malus, 150)     AS bonus_malus_c,
       f.density,
       f.veh_brand, f.veh_gas, f.region,
       CASE WHEN f.driv_age < 25 THEN '1: 18-24'
            WHEN f.driv_age < 30 THEN '2: 25-29'
            WHEN f.driv_age < 40 THEN '3: 30-39'
            WHEN f.driv_age < 50 THEN '4: 40-49'
            WHEN f.driv_age < 60 THEN '5: 50-59'
            WHEN f.driv_age < 70 THEN '6: 60-69'
            ELSE '7: 70+' END      AS driv_age_band,
       CASE WHEN f.veh_age = 0  THEN '1: 0'
            WHEN f.veh_age <= 2 THEN '2: 1-2'
            WHEN f.veh_age <= 5 THEN '3: 3-5'
            WHEN f.veh_age <= 10 THEN '4: 6-10'
            WHEN f.veh_age <= 15 THEN '5: 11-15'
            ELSE '6: 16+' END      AS veh_age_band,
       CASE WHEN f.bonus_malus = 50  THEN '1: 50'
            WHEN f.bonus_malus <= 60 THEN '2: 51-60'
            WHEN f.bonus_malus <= 80 THEN '3: 61-80'
            WHEN f.bonus_malus <= 100 THEN '4: 81-100'
            ELSE '5: 101+' END     AS bm_band,
       CASE f.idpol % 5 WHEN 0 THEN 'test' WHEN 1 THEN 'valid' ELSE 'train' END AS split
FROM clean_freq f
JOIN policy_sev s USING (idpol);
