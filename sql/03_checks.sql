-- name: row_counts_through_pipeline
SELECT 'raw_freq' AS stage, COUNT(*) AS n FROM raw_freq
UNION ALL SELECT 'clean_freq (valid, deduped)', COUNT(*) FROM clean_freq
UNION ALL SELECT 'raw_sev', COUNT(*) FROM raw_sev
UNION ALL SELECT 'clean_sev (amount > 0)', COUNT(*) FROM clean_sev
UNION ALL SELECT 'model_severity (policies with claims)', COUNT(*) FROM model_severity;

-- name: rows_flagged_by_rule
SELECT SUM(exposure <= 0)                      AS exposure_nonpositive,
       SUM(exposure > 1)                       AS exposure_gt_1_capped,
       SUM(driv_age NOT BETWEEN 18 AND 120)    AS driver_age_invalid,
       SUM(veh_age < 0)                        AS veh_age_negative,
       SUM(veh_age > 20)                       AS veh_age_gt_20_capped_later,
       SUM(bonus_malus < 50)                   AS bonus_malus_below_50,
       (SELECT COUNT(*) - COUNT(DISTINCT idpol) FROM stg_freq) AS duplicate_policy_rows
FROM stg_freq;

-- name: severity_rows_with_no_matching_policy
SELECT COUNT(*) AS orphan_claims, COUNT(DISTINCT s.idpol) AS orphan_policies
FROM clean_sev s LEFT JOIN clean_freq f USING (idpol)
WHERE f.idpol IS NULL;

-- name: claimnb_in_freq_vs_rows_in_sev
SELECT SUM(f.claim_nb = s.n_claims) AS agree,
       SUM(f.claim_nb <> s.n_claims) AS disagree
FROM clean_freq f JOIN policy_sev s USING (idpol);

-- name: claim_amount_percentiles
WITH r AS (SELECT claim_amount, NTILE(100) OVER (ORDER BY claim_amount) AS pct FROM clean_sev)
SELECT pct AS percentile, ROUND(MAX(claim_amount), 0) AS claim_amount
FROM r WHERE pct IN (10, 50, 90, 99, 100) GROUP BY pct ORDER BY pct;

-- name: split_sizes
SELECT split, COUNT(*) AS policies, SUM(n_claims) AS claims,
       ROUND(SUM(avg_severity * n_claims) / SUM(n_claims), 0) AS mean_severity
FROM model_severity GROUP BY split ORDER BY split;

-- name: severity_by_driver_age_band
SELECT driv_age_band, COUNT(*) AS policies, SUM(n_claims) AS claims,
       ROUND(SUM(avg_severity * n_claims) / SUM(n_claims), 0) AS mean_severity
FROM model_severity GROUP BY driv_age_band ORDER BY driv_age_band;

-- name: severity_by_bonus_malus_band
SELECT bm_band, COUNT(*) AS policies, SUM(n_claims) AS claims,
       ROUND(SUM(avg_severity * n_claims) / SUM(n_claims), 0) AS mean_severity
FROM model_severity GROUP BY bm_band ORDER BY bm_band;
