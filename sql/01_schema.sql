-- Raw landing tables. Deliberately untyped: we load exactly what the source gives us
-- and do all typing / cleaning in SQL (02_clean.sql), so the cleaning is auditable.
DROP TABLE IF EXISTS raw_freq;
CREATE TABLE raw_freq (
    IDpol, ClaimNb, Exposure, Area, VehPower, VehAge, DrivAge,
    BonusMalus, VehBrand, VehGas, Density, Region
);

DROP TABLE IF EXISTS raw_sev;
CREATE TABLE raw_sev (
    IDpol, ClaimAmount
);
