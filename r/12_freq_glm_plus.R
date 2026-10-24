# Better-specified FREQUENCY GLM ("GLM+"): piecewise-linear splines + two screened interactions.
# Mirrors src/glm_plus.py exactly (same knots, same standardisation, same screening), so the Python check
# and this script should give the same test deviance.
#   main effects : linear + hinge terms max(x - knot, 0) for driver age, bonus-malus, vehicle age, log-density;
#                  bonus-malus == 50 indicator; factors for gas, area, vehicle power, brand, region
#   interactions : top 2 of all 15 pairwise products of six standardised variables, ranked by add1(test = "Rao")
#                  on the fitting data only. Inference uses the Pearson dispersion (quasi-Poisson).

suppressPackageStartupMessages({ library(DBI); library(RSQLite) })
args   <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
root   <- if (length(script) == 1) normalizePath(file.path(dirname(script), "..")) else getwd()
out    <- file.path(root, "outputs"); dir.create(out, showWarnings = FALSE)

con <- dbConnect(SQLite(), file.path(root, "data", "claims.db"))
df  <- dbGetQuery(con, "SELECT * FROM model_frequency ORDER BY idpol")
dbDisconnect(con)

df$log_density <- log(df$density)
df$gas_diesel  <- as.numeric(df$veh_gas == "Diesel")
df$bm50        <- as.numeric(df$bonus_malus_c == 50)
cat_vars <- c("veh_gas", "area", "veh_power_g", "veh_brand", "region")
with_base <- function(x, w) { x <- factor(x); relevel(x, ref = names(which.max(tapply(w, x, sum)))) }
for (v in cat_vars) df[[v]] <- with_base(df[[v]], df$exposure)

knots <- list(driv_age_c = c(21, 25, 30, 40, 50, 60, 70), bonus_malus_c = c(60, 70, 80, 90, 100, 120),
              veh_age_c = c(1, 2, 4, 7, 11, 15), log_density = c(3, 4.5, 6, 7.5, 9))
ix_vars <- c("driv_age_c", "bonus_malus_c", "veh_age_c", "log_density", "veh_power", "gas_diesel")

hinge_names <- character(0)
for (v in names(knots)) for (k in knots[[v]]) {
  nm <- paste0("h_", v, "_", k); df[[nm]] <- pmax(df[[v]] - k, 0); hinge_names <- c(hinge_names, nm)
}

fit_df  <- droplevels(subset(df, split %in% c("train", "valid")))
test_df <- subset(df, split == "test")

# standardise the interaction variables with FITTING-data mean / sd, then form all pair products
pairs <- combn(ix_vars, 2, simplify = FALSE)
cand  <- character(0)
for (v in ix_vars) {
  m <- mean(fit_df[[v]]); s <- sd(fit_df[[v]])
  fit_df[[paste0("z_", v)]]  <- (fit_df[[v]]  - m) / s
  test_df[[paste0("z_", v)]] <- (test_df[[v]] - m) / s
}
for (p in pairs) {
  nm <- paste0("ix_", p[1], "__", p[2]); cand <- c(cand, nm)
  fit_df[[nm]]  <- fit_df[[paste0("z_", p[1])]]  * fit_df[[paste0("z_", p[2])]]
  test_df[[nm]] <- test_df[[paste0("z_", p[1])]] * test_df[[paste0("z_", p[2])]]
}
for (v in cat_vars) {                      # unseen test levels -> base level
  lv <- levels(fit_df[[v]]); x <- as.character(test_df[[v]]); x[!(x %in% lv)] <- lv[1]
  test_df[[v]] <- factor(x, levels = lv)
}

main  <- c(names(knots), hinge_names, "bm50", cat_vars)
base  <- glm(as.formula(paste("claim_nb ~", paste(main, collapse = " + "), "+ offset(log(exposure))")),
             family = poisson(link = "log"), data = fit_df)
sc    <- add1(base, scope = as.formula(paste("~ . +", paste(cand, collapse = " + "))), test = "Rao")
sc    <- sc[rownames(sc) != "<none>", ]
top   <- rownames(sc)[order(-sc[["Rao score"]])][1:2]
glm_p <- update(base, as.formula(paste(". ~ . +", paste(top, collapse = " + "))))

pois_dev <- function(y, mu) mean(2 * (ifelse(y > 0, y * log(y / mu), 0) - (y - mu)))
pred <- predict(glm_p, newdata = test_df, type = "response")
phi  <- sum(residuals(glm_p, type = "pearson")^2) / df.residual(glm_p)
cat(sprintf("GLM+ test mean Poisson deviance: %.5f | Pearson dispersion %.3f\n", pois_dev(test_df$claim_nb, pred), phi))
cat("interactions chosen:", paste(top, collapse = ", "), "\n")

write.csv(data.frame(idpol = test_df$idpol, y = test_df$claim_nb, exposure = test_df$exposure, pred = pred),
          file.path(out, "freq_glm_plus_r_test_preds.csv"), row.names = FALSE)
write.csv(data.frame(interaction = rownames(sc), rao_score = sc[["Rao score"]]),
          file.path(out, "freq_glm_plus_interaction_scores.csv"), row.names = FALSE)
cat("wrote outputs/freq_glm_plus_r_test_preds.csv, freq_glm_plus_interaction_scores.csv\n")
