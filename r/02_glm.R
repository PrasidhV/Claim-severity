# Baseline severity GLM, the way a pricing actuary would build it:
#   * Gamma family, log link  -> multiplicative relativities
#   * response = average claim cost per policy, prior weights = number of claims
#   * banded rating factors (age, vehicle age, bonus-malus), base level = most-populated level
# Reads the SQL-built table directly from SQLite. Packages: DBI, RSQLite.

suppressPackageStartupMessages({ library(DBI); library(RSQLite) })

args   <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
root   <- if (length(script) == 1) normalizePath(file.path(dirname(script), "..")) else getwd()
out    <- file.path(root, "outputs"); dir.create(out, showWarnings = FALSE)

con <- dbConnect(SQLite(), file.path(root, "data", "claims.db"))
df  <- dbGetQuery(con, "SELECT * FROM model_severity ORDER BY idpol")
dbDisconnect(con)

df$log_density <- log(df$density)
cat_vars <- c("driv_age_band", "veh_age_band", "bm_band", "veh_gas",
              "area", "veh_power_g", "veh_brand", "region")

# base level = level carrying the most claims (stable, well-estimated reference)
with_base <- function(x, w) {
  x <- factor(x)
  relevel(x, ref = names(which.max(tapply(w, x, sum))))
}
for (v in cat_vars) df[[v]] <- with_base(df[[v]], df$n_claims)

fit_df  <- droplevels(subset(df, split %in% c("train", "valid")))   # same data the GBM sees (train + valid)
test_df <- subset(df, split == "test")

form <- avg_severity ~ driv_age_band + veh_age_band + bm_band + veh_gas +
        area + veh_power_g + veh_brand + region + log_density
glm_fit <- glm(form, family = Gamma(link = "log"), data = fit_df, weights = n_claims)

# test levels never seen in fitting fall back to the base level
for (v in cat_vars) {
  lv <- levels(fit_df[[v]])
  x  <- as.character(test_df[[v]])
  x[!(x %in% lv)] <- lv[1]
  test_df[[v]] <- factor(x, levels = lv)
}

gamma_dev <- function(y, mu, w) sum(2 * w * ((y - mu) / mu - log(y / mu))) / sum(w)

pred <- predict(glm_fit, newdata = test_df, type = "response")
cat(sprintf("GLM  test weighted mean Gamma deviance: %.4f\n",
            gamma_dev(test_df$avg_severity, pred, test_df$n_claims)))
cat(sprintf("GLM  estimated dispersion: %.3f\n", summary(glm_fit)$dispersion))

write.csv(data.frame(idpol = test_df$idpol, y = test_df$avg_severity,
                     w = test_df$n_claims, pred = pred),
          file.path(out, "glm_r_test_preds.csv"), row.names = FALSE)

# Relativities = exp(coef): "this level costs x% more/less per claim than the base level"
est <- coef(summary(glm_fit)); ci <- confint.default(glm_fit)
rel <- data.frame(term = rownames(est), relativity = exp(est[, "Estimate"]),
                  lo = exp(ci[, 1]), hi = exp(ci[, 2]), p_value = est[, "Pr(>|t|)"])
write.csv(rel, file.path(out, "glm_relativities.csv"), row.names = FALSE)

# Which rating factors earn their keep? (F-test for dropping each term)
capture.output(summary(glm_fit), drop1(glm_fit, test = "F"),
               file = file.path(out, "glm_summary.txt"))
cat("wrote outputs/glm_r_test_preds.csv, glm_relativities.csv, glm_summary.txt\n")
