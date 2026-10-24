-- =====================================================================
-- 04_frequency.sql : one row per VALID POLICY (claimants and non-claimants) for the frequency model.
-- Target  : claim_nb (claim count), with exposure (years) as a Poisson offset.
-- ClaimNb is capped at 4: a handful of policies have more, and they would dominate a Poisson fit.
-- Same features, bands and idpol % 5 split as model_severity, so the two models are comparable.
-- =====================================================================
DROP TABLE IF EXISTS model_frequency;
CREATE TABLE model_frequency AS
SELECT f.idpol,
       MIN(f.claim_nb, 4)          AS claim_nb,
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
FROM clean_freq f;
