# ----
# Customer Churn - Survival Analysis in R (Phase 7)
# Kaplan-Meier by segment + log-rank + Cox proportional hazards.
#
# Developed to run in RStudio (Source > Source with Ctrl+Enter) or headless:
#   LANG=en_US.UTF-8 LC_ALL= "C:/Program Files/R/R-4.6.0/bin/Rscript" \
#       r/survival_analysis.R
# Run from the project root, or just let the script find it.
#
# Every table it prints is also written to outputs/r/ as CSV, and the console
# transcript is saved to outputs/r/r_survival_log.txt, so the R numbers are
# inspectable rather than "trust me". outputs/r/run_manifest.txt records the R
# and package versions that produced them.
#
# LIMIT: Telco data is a snapshot. tenure is time observed at the snapshot, and
# Contract is the contract observed at the snapshot - not necessarily the
# contract at signup. KM by contract is therefore descriptive, not a causal
# contract effect.
# ----

# ---- Locate the project root ----
find_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
    candidate <- dirname(script_dir)
    if (file.exists(file.path(candidate, "data", "processed", "clean_telco.csv"))) return(candidate)
  }
  if (file.exists("data/processed/clean_telco.csv")) return(normalizePath("."))
  stop("run from the project root (the folder containing data/processed/clean_telco.csv)")
}
root <- find_root()
out_dir <- file.path(root, "outputs", "r")
fig_dir <- file.path(root, "outputs", "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
con <- file(file.path(out_dir, "r_survival_log.txt"), open = "wt", encoding = "UTF-8")
sink(con, split = TRUE)
on.exit({
  sink()
  close(con)
}, add = TRUE)

# ---- Load libraries ----
suppressPackageStartupMessages({
  library(dplyr)
  library(survival)
  library(ggplot2)
})

cat("R version:", R.version.string, "\n")
cat("survival:", as.character(packageVersion("survival")),
    "| dplyr:", as.character(packageVersion("dplyr")),
    "| ggplot2:", as.character(packageVersion("ggplot2")), "\n")
writeLines(c(
  paste("run_at_utc:", format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  paste("R_version:", R.version.string),
  paste("survival:", as.character(packageVersion("survival"))),
  paste("dplyr:", as.character(packageVersion("dplyr"))),
  paste("ggplot2:", as.character(packageVersion("ggplot2"))),
  paste("script: r/survival_analysis.R"),
  "note: KM by contract is descriptive; contract is observed at snapshot, not at signup."
), file.path(out_dir, "run_manifest.txt"))

# ---- Load data ----
telco <- read.csv(file.path(root, "data", "processed", "clean_telco.csv")) |>
  mutate(
    event = as.integer(Churn == "Yes"),
    duration = as.numeric(tenure)
  )

cat("\nRows:", nrow(telco), " | churn rate:", round(mean(telco$event), 4),
    " | duration range:", min(telco$duration), "-", max(telco$duration), "\n")
stopifnot(identical(as.integer(telco$event), as.integer(telco$target)))

times <- c(12, 24, 36, 48, 60, 72)

# pchisq underflows to exactly 0 for these enormous log-rank statistics; keep it readable.
fmt_p <- function(p) ifelse(p < 1e-16, "< 1e-16", formatC(p, format = "g", digits = 4))

# ---- Helper: tidy a survfit into a CSV ----
km_summary_csv <- function(fit, segment_type, path) {
  s <- summary(fit, times = times, censored = TRUE)
  df <- data.frame(
    segment_type = segment_type,
    segment = as.character(s$strata),
    month = as.numeric(s$time),
    n_at_risk = as.integer(s$n.risk),
    n_event = as.integer(s$n.event),
    survival_probability = round(as.numeric(s$surv), 4),
    std_err = round(as.numeric(s$std.err), 4),
    ci_lower = round(as.numeric(s$lower), 4),
    ci_upper = round(as.numeric(s$upper), 4)
  )
  df$segment <- sub("^[^=]*=", "", df$segment)
  write.csv(df, path, row.names = FALSE)
  df
}

# ---- Kaplan-Meier by contract ----
cat("\n==== Kaplan-Meier by contract ====\n")
fit_contract <- survfit(Surv(duration, event) ~ Contract, data = telco)
print(fit_contract)
print(summary(fit_contract, times = times))

km_contract <- km_summary_csv(fit_contract, "Contract", file.path(out_dir, "km_summary_contract.csv"))
cat("\nWritten: outputs/r/km_summary_contract.csv\n")

med_contract <- data.frame(
  segment_type = "Contract",
  segment = names(fit_contract$strata) |> sub("^[^=]*=", "", x = _),
  median_survival_months = as.numeric(summary(fit_contract)$table[, "median"])
)
write.csv(med_contract, file.path(out_dir, "km_median_contract.csv"), row.names = FALSE)
print(med_contract)

logrank_contract <- survdiff(Surv(duration, event) ~ Contract, data = telco)
lr_contract <- data.frame(
  test = "log-rank",
  segment_type = "Contract",
  chisq = round(logrank_contract$chisq, 3),
  df = length(logrank_contract$n) - 1,
  p_value = fmt_p(pchisq(logrank_contract$chisq, length(logrank_contract$n) - 1, lower.tail = FALSE))
)
cat("\n==== Log-rank, survival by contract ====\n")
print(logrank_contract)
print(lr_contract)
write.csv(lr_contract, file.path(out_dir, "logrank_contract.csv"), row.names = FALSE)

# ---- Kaplan-Meier by payment method ----
cat("\n==== Kaplan-Meier by payment method ====\n")
fit_payment <- survfit(Surv(duration, event) ~ PaymentMethod, data = telco)
km_payment <- km_summary_csv(fit_payment, "PaymentMethod", file.path(out_dir, "km_summary_payment.csv"))
print(summary(fit_payment, times = times))
logrank_payment <- survdiff(Surv(duration, event) ~ PaymentMethod, data = telco)
lr_payment <- data.frame(
  test = "log-rank", segment_type = "PaymentMethod",
  chisq = round(logrank_payment$chisq, 3),
  df = length(logrank_payment$n) - 1,
  p_value = fmt_p(pchisq(logrank_payment$chisq, length(logrank_payment$n) - 1, lower.tail = FALSE))
)
print(lr_payment)
write.csv(lr_payment, file.path(out_dir, "logrank_payment.csv"), row.names = FALSE)

# ---- KM figures (ggplot2 only: no survminer dependency) ----
# summary.survfit returns the step function aligned across strata, which is the
# tidy source for a ggplot step curve (fit$surv alone is a times x strata matrix).
surv_df <- function(fit) {
  s <- summary(fit)
  data.frame(
    segment = sub("^[^=]*=", "", as.character(s$strata)),
    time = as.numeric(s$time),
    surv = as.numeric(s$surv),
    lower = as.numeric(s$lower),
    upper = as.numeric(s$upper)
  )
}
p1 <- ggplot(surv_df(fit_contract), aes(x = time, y = surv, colour = segment)) +
  geom_step(linewidth = 0.9) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = segment), alpha = 0.12, colour = NA) +
  labs(title = "Kaplan-Meier survival by contract (R survival package)",
       subtitle = "duration = tenure; event = Churn == 'Yes'. Descriptive, not causal.",
       x = "tenure (months)", y = "survival probability", colour = "Contract", fill = "Contract") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "km_by_contract_R.png"), p1, width = 9, height = 5.2, dpi = 150)
cat("\nWritten: outputs/figures/km_by_contract_R.png\n")

p2 <- ggplot(surv_df(fit_payment), aes(x = time, y = surv, colour = segment)) +
  geom_step(linewidth = 0.9) +
  labs(title = "Kaplan-Meier survival by payment method",
       x = "tenure (months)", y = "survival probability", colour = "PaymentMethod") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "km_by_payment_R.png"), p2, width = 9, height = 5.2, dpi = 150)
cat("Written: outputs/figures/km_by_payment_R.png\n")

# ---- Cox proportional hazards model ----
cat("\n==== Cox proportional hazards ====\n")
cox_model <- coxph(
  Surv(duration, event) ~ MonthlyCharges + SeniorCitizen + service_count +
    contract_risk + electronic_check_flag + no_support_flag + bill_shock_flag,
  data = telco
)
print(summary(cox_model))

# summary.coxph() splits the output: $coefficients has coef/exp(coef)/se/z/p and
# the confidence intervals live in a separate $conf.int matrix. conf.int is
# already on the hazard-ratio scale, so the bounds are used as-is, not re-exponentiated.
cs <- summary(cox_model)
cox_table <- cbind(
  as.data.frame(cs$coefficients),
  as.data.frame(cs$conf.int)[, c("lower .95", "upper .95")]
) |>
  mutate(
    covariate = rownames(cs$coefficients),
    hr_lower95 = `lower .95`,
    hr_upper95 = `upper .95`
  ) |>
  select(covariate, coef, `exp(coef)`, `se(coef)`, hr_lower95, hr_upper95, z, `Pr(>|z|)`) |>
  rename(hazard_ratio = `exp(coef)`, std_err = `se(coef)`, p_value = `Pr(>|z|)`) |>
  mutate(across(c(coef, hazard_ratio, std_err, hr_lower95, hr_upper95, z), ~ round(.x, 6)),
         # round(p, 6) would flatten the sub-1e-40 p-values to 0 and destroy the
         # Python/R agreement check, so keep significant digits instead.
         p_value = signif(p_value, 4))
cat("\nHazard ratios:\n")
print(cox_table)
write.csv(cox_table, file.path(out_dir, "cox_hazard_ratios.csv"), row.names = FALSE)
cat("Written: outputs/r/cox_hazard_ratios.csv\n")

# summary.coxph names this component `nevent` (not `n.event`) and `sctest` is the
# ordered triple c(chisq, df, p).
st <- as.numeric(cs$sctest)
model_fit <- data.frame(
  n = cs$n, events = cs$nevent,
  log_likelihood = round(cs$loglik[2], 3),
  concordance = round(cs$concordance[1], 4),
  concordance_se = round(cs$concordance[2], 4),
  global_test = "Likelihood ratio",
  global_chisq = round(st[1], 3),
  global_df = st[2],
  global_p = fmt_p(st[3])
)
cat("\nModel fit:\n")
print(model_fit)
write.csv(model_fit, file.path(out_dir, "cox_model_fit.csv"), row.names = FALSE)

# ---- Proportional-hazards assumption (Schoenfeld residuals) ----
cat("\n==== Schoenfeld proportional-hazards test ====\n")
# transform="identity" keeps the residual x-axis in tenure months instead of the
# default Kaplan-Meier probability scale.
zph <- cox.zph(cox_model, transform = "identity")
# cox.zph()$table columns are chisq / df / p (survival 3.x); GLOBAL is a row.
zph_df <- as.data.frame(zph$table) |>
  mutate(term = rownames(zph$table),
         p_value = fmt_p(p),
         across(c(chisq, p), ~ round(.x, 5))) |>
  select(term, chisq, df, p, p_value)
print(zph_df)
write.csv(zph_df, file.path(out_dir, "cox_zph_test.csv"), row.names = FALSE)
cat("Written: outputs/r/cox_zph_test.csv\n")

# Scaled Schoenfeld residuals: $y is a records x covariates matrix, $time the
# matching event times. Points are noisy at this n, so a loess smoother carries
# the signal a flat-vs-sloped read depends on.
resid_df <- as.data.frame(zph$y)
resid_df$time <- as.numeric(zph$time)
covariates <- setdiff(names(resid_df), "time")
# base-R long reshape: tidyr is not a dependency of this script.
long <- do.call(rbind, lapply(covariates, function(v) {
  data.frame(covariate = v, time = resid_df$time,
             resid = as.numeric(resid_df[[v]]), stringsAsFactors = FALSE)
}))
p3 <- ggplot(long, aes(x = time, y = resid)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey40") +
  geom_point(alpha = 0.08, size = 0.6, colour = "grey50") +
  geom_smooth(method = "loess", se = TRUE, colour = "#1f4e79", linewidth = 0.9) +
  facet_wrap(~ covariate, scales = "free_y") +
  labs(title = "Scaled Schoenfeld residuals (flat = proportional hazards holds)",
       subtitle = paste("GLOBAL test chi-sq =", signif(zph$table["GLOBAL", "chisq"], 4),
                        "| p =", fmt_p(zph$table["GLOBAL", "p"]))
       , x = "time (months)", y = "scaled residual") +
  theme_minimal(base_size = 10)
ggsave(file.path(fig_dir, "cox_schoenfeld_R.png"), p3, width = 11, height = 7, dpi = 150)
cat("Written: outputs/figures/cox_schoenfeld_R.png\n")

cat("\nDone. Interpretation: hazard_ratio > 1 = higher churn hazard, < 1 = lower.\n")
