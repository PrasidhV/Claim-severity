# Baseline FREQUENCY GLM, the way a pricing actuary would build it:
#   * Poisson family, log link, offset(log(exposure))  -> multiplicative relativities on claims per year
#   * banded rating factors; base level = level with the most exposure
#   * inference is quasi-Poisson: standard errors and F-tests use the PEARSON dispersion, because
#     R's poisson summary() silently fixes the dispersion at 1 (it is not exactly 1 on real data).
# Reads the SQL-built table directly from SQLite. Packages: DBI, RSQLite.

suppressPackageStartupMessages({ library(DBI); library(RSQLite) })

args   <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
root   <- if (length(script) == 1) normalizePath(file.path(dirname(script), "..")) else getwd()
out    <- file.path(root, "outputs"); dir.create(out, showWarnings = FALSE)

con <- dbConnect(SQLite(), file.path(root, "data", "claims.db"))
df  <- dbGetQuery(con, "SELECT * FROM model_frequency ORDER BY idpol")
dbDisconnect(con)

df$log_density <- log(df$density)
cat_vars <- c("driv_age_band", "veh_age_band", "bm_band", "veh_gas",
              "area", "veh_power_g", "veh_brand", "region")

with_base <- function(x, w) {
  x <- factor(x)
  relevel(x, ref = names(which.max(tapply(w, x, sum))))
}
for (v in cat_vars) df[[v]] <- with_base(df[[v]], df$exposure)

fit_df  <- droplevels(subset(df, split %in% c("train", "valid")))
test_df <- subset(df, split == "test")

form <- claim_nb ~ driv_age_band + veh_age_band + bm_band + veh_gas +
        area + veh_power_g + veh_brand + region + log_density + offset(log(exposure))
glm_fit <- glm(form, family = poisson(link = "log"), data = fit_df)

for (v in cat_vars) {                      # unseen test levels fall back to the base level
  lv <- levels(fit_df[[v]])
  x  <- as.character(test_df[[v]])
  x[!(x %in% lv)] <- lv[1]
  test_df[[v]] <- factor(x, levels = lv)
}

pois_dev <- function(y, mu) mean(2 * (ifelse(y > 0, y * log(y / mu), 0) - (y - mu)))

pred <- predict(glm_fit, newdata = test_df, type = "response")     # predicted COUNTS (offset included)
phi  <- sum(residuals(glm_fit, type = "pearson")^2) / df.residual(glm_fit)
cat(sprintf("FREQ GLM test mean Poisson deviance: %.5f\n", pois_dev(test_df$claim_nb, pred)))
cat(sprintf("FREQ GLM Pearson dispersion: %.3f\n", phi))

write.csv(data.frame(idpol = test_df$idpol, y = test_df$claim_nb,
                     exposure = test_df$exposure, pred = pred),
          file.path(out, "freq_glm_r_test_preds.csv"), row.names = FALSE)

# Relativities with quasi-Poisson (dispersion-scaled) confidence intervals
b  <- coef(glm_fit); b <- b[!is.na(b)]
se <- sqrt(diag(vcov(glm_fit))) * sqrt(phi)
rel <- data.frame(term = names(b), relativity = exp(b),
                  lo = exp(b - 1.96 * se[names(b)]), hi = exp(b + 1.96 * se[names(b)]),
                  p_value = 2 * pnorm(-abs(b / se[names(b)])))
write.csv(rel, file.path(out, "freq_glm_relativities.csv"), row.names = FALSE)

# Per-term F-tests with the Pearson dispersion (same fix as the severity GLM)
tt <- do.call(rbind, lapply(attr(terms(glm_fit), "term.labels"), function(t) {
  small <- update(glm_fit, as.formula(paste(". ~ . -", t)))
  ddf   <- df.residual(small) - df.residual(glm_fit)
  Fval  <- ((deviance(small) - deviance(glm_fit)) / ddf) / phi
  data.frame(term = t, df = ddf, F = Fval,
             p = pf(Fval, ddf, df.residual(glm_fit), lower.tail = FALSE))
}))
write.csv(tt, file.path(out, "freq_glm_term_tests.csv"), row.names = FALSE)
capture.output(summary(glm_fit), tt, file = file.path(out, "freq_glm_summary.txt"))
print(tt, digits = 3)
cat("wrote outputs/freq_glm_r_test_preds.csv, freq_glm_relativities.csv, freq_glm_term_tests.csv, freq_glm_summary.txt\n")
