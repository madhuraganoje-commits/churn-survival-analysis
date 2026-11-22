# Who Leaves, and When? Predicting Customer Churn with Survival Analysis

**Madhura Ganoje** | MSc Business Analytics, Queen Mary University of London | 2026

Python (pandas, lifelines, scikit-survival, scikit-learn, matplotlib, seaborn) | 7,043 customers | 3 models

---

## In one minute

- **The problem:** a telecoms provider loses about 1 in 4 customers. A retention team needs to know who is at risk and when, so it can step in before the customer decides to leave.
- **The approach:** instead of a yes/no churn classifier, I used survival analysis, which models time until churn. I built three models in sequence, each one checking and extending the one before.
- **The result:** a model that ranks customers by how soon they are likely to leave (test concordance 0.837), and four risk tiers whose real churn rates run from 1.8% to 61.3%.
- **The insight:** contract type matters more than anything else. Customers on two-year contracts have a 92% lower churn hazard than month-to-month customers.

---

## 1. Framing the problem

A yes/no churn model answers "will this customer leave?" A retention team needs more than that. It needs to know **when**, so it can decide who to contact first.

This led to two framing decisions:

1. **Treat churn as a time-to-event problem.** Each customer has a tenure (months with the provider) and either has left or is still active. For customers who are still active, we only know they have not left *yet*. Survival analysis handles this properly (it treats them as *censored*), while a classifier would treat them as a fixed "no".
2. **Make the output usable by non-technical teams.** The end product should be a short list of risk groups a CRM team can act on, not a table of coefficients.

## 2. The data

- **Source:** IBM Telco Customer Churn dataset (public), 7,043 residential customers, 21 columns.
- **Features:** demographics (e.g. senior citizen, partner), services (internet type, online security, tech support), and account details (contract type, payment method, monthly charges).
- **Target:** churned (yes/no), with tenure as the time variable. 1,869 customers (26.5%) churned.

## 3. Cleaning and preparing the data

| Issue found | What I did | Why |
|---|---|---|
| `TotalCharges` stored as text, with 11 blanks | Converted to numeric; set the 11 blanks to 0 | All 11 were brand-new customers (tenure 0) who had not been billed yet |
| `TotalCharges` almost equals tenure x monthly charges (r = 0.83 with tenure) | Dropped it from the models | Keeping it would double-count tenure and make model coefficients unstable |
| 15 categorical columns | One-hot encoded, dropping one level each | Avoids perfectly overlapping columns in the regression model |
| Tenure of 0 months | Set to a minimum of 0.5 months | Survival models need a positive time value |
| Monthly charges on a different scale | Standardised for the Cox model | Makes coefficients comparable |

No duplicates or other missing values were found.

## 4. Exploring the data

Before modelling, I looked at churn rates across each feature.

![Churn rate by key features](images/01_churn_by_feature.png)

What stood out:

- **Contract type:** 42.7% of month-to-month customers churned, against 2.8% on two-year contracts.
- **Internet service:** fiber optic customers churned at 41.9%, more than double DSL (19.0%).
- **Payment method:** electronic check users churned at 45.3%.
- **Timing:** churn is heavily front-loaded. 47% of customers in their first year churned, falling below 7% after four years.

That last point is exactly what a yes/no classifier would miss, and it confirmed the survival framing.

## 5. Modelling: three models, each with a job

| Model | What it does | Why I used it |
|---|---|---|
| **Kaplan-Meier** | Draws a retention curve over time for each group | A simple, transparent baseline that anyone can read |
| **Cox proportional hazards** | Regression that estimates how much each factor raises or lowers churn risk, holding the others constant | Separates effects that overlap (e.g. fiber customers also tend to be on monthly contracts) |
| **Random Survival Forest** | A random forest adapted for time-to-event data | Captures non-linear effects and interactions, and does not rely on the Cox model's assumptions |

### Step 1: Kaplan-Meier, to see the pattern

![Retention by contract type](images/02_km_by_contract.png)

The gap between month-to-month and fixed-term customers is the largest in the data (log-rank test p < 0.0001). But contract, internet type and payment method are correlated with each other, so this view cannot say which one actually drives churn. That is what the next model is for.

### Step 2: Cox model, to separate the drivers

![Cox hazard ratios](images/03_cox_hazard_ratios.png)

A hazard ratio below 1 means lower churn risk; above 1 means higher.

- Two-year contract: **92% lower** churn hazard (one-year: 76% lower)
- Online security / tech support: **35-48% lower**
- Electronic check payment: **75% higher**
- Fiber optic internet: **47% higher**, even after accounting for price and contract

**Checking the model's assumptions.** The Cox model assumes each factor's effect stays constant over time. I tested this (Schoenfeld residual tests) and found nine variables where it does not hold, mostly add-on services whose protective effect fades over time. The key contract, payment and internet results passed. This finding was my main reason for adding a third, more flexible model.

### Step 3: Random Survival Forest, to build a robust predictor

- Split the data 80/20 (5,634 training, 1,409 test customers).
- Tuned it with 5-fold cross-validated grid search over 12 settings (number of trees, leaf size, features per split).
- **Test concordance: 0.837.** In plain terms, if you pick two customers at random, the model correctly predicts which one leaves first 84% of the time (random guessing: 50%).
- **Integrated Brier score: 0.096** (0 is perfect, 0.25 is uninformative), so its probability estimates are well calibrated as well as well ranked.
- The train/test gap was small (0.872 vs 0.837), so it is not badly overfitting.

**An honest comparison.** On the same held-out test set the Cox model scored slightly higher (0.851). I still preferred the forest for scoring customers because it does not depend on the assumption that the Cox model partly broke. Both models agreed on the main drivers, which gives me more confidence in the findings.

## 6. From model to insight

**Customer profiles.** I asked the model to predict retention for three typical customers:

![Predicted retention for three customer profiles](images/04_customer_profiles.png)

A month-to-month, fiber optic, electronic check customer has only a **30% chance of still being a customer after 12 months**. A two-year, DSL, auto-pay customer: **98%**.

**Risk tiers.** I scored all 7,043 customers and split them into four equal groups. To check the scores mean something, I plotted the groups' *actual* retention:

![Actual retention by model risk tier](images/05_risk_segments.png)

| Risk tier | Customers | Actual churn rate | Avg tenure (months) |
|---|---|---|---|
| Low | 1,761 | 1.8% | 53.5 |
| Medium | 1,761 | 10.7% | 39.2 |
| High | 1,760 | 32.3% | 23.8 |
| Very high | 1,761 | 61.3% | 13.0 |

The tiers separate cleanly, so a retention team can use them directly.

## 7. Recommendations

1. **Make contract conversion the main retention lever.** Offer time-limited incentives to month-to-month customers in the very high risk tier.
2. **Contact the very high risk tier early,** within the first 6 months, prioritised by each customer's predicted retention curve.
3. **Investigate fiber optic service quality.** Its extra risk remains after accounting for price, which suggests a service or value issue. Survey these customers before offering discounts.
4. **Move electronic check users to automatic payment** with a small bill credit.
5. **Bundle online security and tech support** into standard packages for high-risk customers.

Matching actions to tiers: low risk needs no action; medium gets digital nudges; high gets a proactive call; very high goes to retention specialists.

## 8. Limitations and what I would do next

- **The model predicts risk; it does not prove a retention offer will work.** The next step would be a randomised test: give offers to a random half of high-risk customers and compare churn with the other half.
- **One provider, one point in time.** The model would need retraining on live data and regular checks.
- **Missing signals.** No satisfaction scores, complaints, outages or competitor prices, which would likely improve the model.
- **Time-varying effects.** A Cox model with time-varying effects, or a neural network survival model such as DeepSurv, could handle the broken assumption more directly.
- **Productising it:** wrap the scoring in a simple app where a retention manager picks a customer and sees their predicted retention curve and tier.

## 9. Repository structure

```
churn-survival-analysis/
  README.md                  this case study
  images/                    figures used above
  notebooks/
    01_eda.ipynb             exploration and cleaning
    02_kaplan_meier.ipynb    retention curves and log-rank tests
    03_cox_model.ipynb       hazard ratios and assumption checks
    04_random_survival_forest.ipynb   tuning, evaluation, profiles, risk tiers
  data/                      instructions to download the IBM Telco dataset
```

## Tools

Python: pandas, NumPy, lifelines, scikit-survival, scikit-learn (train/test split, GridSearchCV, permutation importance), matplotlib, seaborn. Jupyter.

*Originally completed as an individual project for Masterclass in Business Analytics (Machine Learning), Queen Mary University of London.*
