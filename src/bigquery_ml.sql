-- How the reactivation model would run in production with BigQuery ML.
-- Assumes the star schema from Initiative 1 and a feature view built with dbt:
-- one row per inactive seller per month, with the 3-month activation label.

-- 1. Train (months with a complete 3-month label window)
CREATE OR REPLACE MODEL `sellers.reactivation_model`
OPTIONS (
  model_type = 'logistic_reg',
  input_label_cols = ['label'],
  auto_class_weights = FALSE
) AS
SELECT
  tenure_months, logins_last_3m, LN(1 + listings) AS log_listings,
  ever_sold, months_inactive, onboarding_steps,
  category, country, business_unit, onboarding_channel, size_band,
  label
FROM `sellers.inactive_seller_snapshots`
WHERE snapshot_month BETWEEN 12 AND 30;

-- 2. Evaluate on later months the model never saw
SELECT *
FROM ML.EVALUATE(MODEL `sellers.reactivation_model`, (
  SELECT tenure_months, logins_last_3m, LN(1 + listings) AS log_listings,
         ever_sold, months_inactive, onboarding_steps,
         category, country, business_unit, onboarding_channel, size_band, label
  FROM `sellers.inactive_seller_snapshots`
  WHERE snapshot_month BETWEEN 32 AND 34));

-- 3. Score today's inactive sellers every month; Looker reads this table
CREATE OR REPLACE TABLE `sellers.reactivation_scores` AS
SELECT seller_id, predicted_label_probs[OFFSET(0)].prob AS score
FROM ML.PREDICT(MODEL `sellers.reactivation_model`, (
  SELECT seller_id, tenure_months, logins_last_3m, LN(1 + listings) AS log_listings,
         ever_sold, months_inactive, onboarding_steps,
         category, country, business_unit, onboarding_channel, size_band
  FROM `sellers.inactive_seller_snapshots`
  WHERE snapshot_month = 36));
