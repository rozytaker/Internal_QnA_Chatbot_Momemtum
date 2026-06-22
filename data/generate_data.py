"""
Momentum Africa — UC05 Synthetic Data Generator
=================================================
Creates the 16 tables referenced in the discovery deck (slide: "UC05 — Data
Layer & Example Queries") so the Q&A Copilot has something realistic to
query: claims, policy, Multiply and distribution data across South Africa,
Lesotho, Botswana and Namibia.

This is illustrative data for demo purposes only — not real client data.
Run once before starting the app:

    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).parent
TODAY = pd.Timestamp.now().normalize()
START = TODAY - pd.DateOffset(months=24)
MONTHS = pd.date_range(START, TODAY, freq="MS")  # month starts, 24-25 months

MARKETS = ["South Africa", "Lesotho", "Botswana", "Namibia"]
MARKET_CODE = {"South Africa": "ZA", "Lesotho": "LSO", "Botswana": "BWA", "Namibia": "NAM"}
MARKET_WEIGHTS = [0.55, 0.13, 0.17, 0.15]  # SA is the largest book

TERRITORIES = {
    "South Africa": ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape"],
    "Lesotho": ["Maseru", "Berea"],
    "Botswana": ["Gaborone", "Francistown"],
    "Namibia": ["Windhoek", "Swakopmund"],
}

PRODUCTS = ["Funeral", "Life", "Health", "Investment"]
PRODUCT_WEIGHTS = [0.42, 0.28, 0.20, 0.10]

CHANNELS = ["Tied Advisor", "Independent Broker", "Bancassurance", "Direct Digital"]
CHANNEL_WEIGHTS = [0.40, 0.27, 0.18, 0.15]

CELLS = ["Motor Cell", "Fleet Cell", "Agri Cell", "Property Cell", "Funeral Cell", "SME Cell"]

BRANDS = {"Funeral": "Metropolitan", "Life": "Momentum", "Health": "Momentum", "Investment": "Momentum"}

LANGS = ["English", "Afrikaans", "Sesotho", "Setswana"]

# Demo storylines deliberately seeded into the data so headline example
# questions from the deck return a clear, presentable answer:
#   - Botswana funeral / Direct Digital channel has elevated lapse
#   - Motor Cell has the highest fraud rate
#   - One territory (Namibia / Swakopmund) shows declining advisor conversion
LAPSE_BIAS = {("Botswana", "Funeral", "Direct Digital"): 1.9}
FRAUD_BIAS_CELL = {"Motor Cell": 2.4, "Agri Cell": 0.6}
DECLINING_TERRITORY = "Swakopmund"


def pick(values, weights, n):
    return rng.choice(values, size=n, p=weights)


def random_dates(start, end, n):
    days = (end - start).days
    offsets = rng.integers(0, max(days, 1), size=n)
    return start + pd.to_timedelta(offsets, unit="D")


# ---------------------------------------------------------------------------
# 1. POLICY MASTER
# ---------------------------------------------------------------------------
N_POLICIES = 9000
policy_id = [f"POL-{i:06d}" for i in range(1, N_POLICIES + 1)]
client_id = [f"CLI-{rng.integers(1, int(N_POLICIES * 0.72)):06d}" for _ in range(N_POLICIES)]  # some clients hold >1 policy
market = pick(MARKETS, MARKET_WEIGHTS, N_POLICIES)
product_type = pick(PRODUCTS, PRODUCT_WEIGHTS, N_POLICIES)
channel = pick(CHANNELS, CHANNEL_WEIGHTS, N_POLICIES)
inception_date = random_dates(START - pd.DateOffset(years=3), TODAY, N_POLICIES)
n_advisors = 160
advisor_id = [f"ADV-{rng.integers(1, n_advisors + 1):03d}" for _ in range(N_POLICIES)]
sum_assured = rng.choice([15000, 30000, 50000, 100000, 250000, 500000], size=N_POLICIES,
                          p=[0.30, 0.25, 0.20, 0.13, 0.08, 0.04])
premium = (sum_assured / rng.uniform(180, 260, N_POLICIES)).round(2)

policy_master = pd.DataFrame({
    "policy_id": policy_id,
    "client_id": client_id,
    "market": market,
    "product_type": product_type,
    "channel": channel,
    "brand": [BRANDS[p] for p in product_type],
    "advisor_id": advisor_id,
    "inception_date": inception_date,
    "sum_assured": sum_assured,
    "premium": premium,
})
policy_master["status"] = "Active"

# ---------------------------------------------------------------------------
# 2. LAPSE EVENTS
# ---------------------------------------------------------------------------
base_lapse_prob = 0.16
lapse_prob = np.full(N_POLICIES, base_lapse_prob)
for i, (m, p, c) in enumerate(zip(market, product_type, channel)):
    bias = LAPSE_BIAS.get((m, p, c))
    if bias:
        lapse_prob[i] *= bias
lapse_prob = np.clip(lapse_prob + rng.normal(0, 0.04, N_POLICIES), 0.02, 0.55)
is_lapsed = rng.random(N_POLICIES) < lapse_prob

lapse_reasons = ["Affordability", "Non-payment", "Dissatisfaction", "Switched provider", "Other"]
lapse_idx = np.where(is_lapsed)[0]
lapse_dates = [inception_date[i] + pd.Timedelta(days=int(rng.integers(90, 900))) for i in lapse_idx]
lapse_dates = [min(d, TODAY) for d in lapse_dates]

lapse_events = pd.DataFrame({
    "policy_id": [policy_id[i] for i in lapse_idx],
    "market": [market[i] for i in lapse_idx],
    "product_type": [product_type[i] for i in lapse_idx],
    "channel": [channel[i] for i in lapse_idx],
    "lapse_date": lapse_dates,
    "lapse_flag": 1,
    "reason": rng.choice(lapse_reasons, size=len(lapse_idx), p=[0.34, 0.30, 0.16, 0.12, 0.08]),
})
policy_master.loc[lapse_idx, "status"] = "Lapsed"

# ---------------------------------------------------------------------------
# 3. NEW BUSINESS  (policies issued within the trailing 24 months)
# ---------------------------------------------------------------------------
nb_mask = inception_date >= START
new_business = pd.DataFrame({
    "policy_id": np.array(policy_id)[nb_mask],
    "market": np.array(market)[nb_mask],
    "product_type": np.array(product_type)[nb_mask],
    "channel": np.array(channel)[nb_mask],
    "advisor_id": np.array(advisor_id)[nb_mask],
    "issue_date": inception_date[nb_mask],
    "premium": premium[nb_mask],
})

# ---------------------------------------------------------------------------
# 4. PREMIUM HISTORY  (last 6 months per active policy)
# ---------------------------------------------------------------------------
hist_months = MONTHS[-6:]
ph_rows = []
for pid, prem, m_idx in zip(policy_id, premium, range(N_POLICIES)):
    on_time_base = rng.uniform(0.75, 0.99)
    for period in hist_months:
        paid_flag = rng.random() < on_time_base
        ph_rows.append((pid, period, prem, prem if paid_flag else 0.0, int(paid_flag)))
premium_history = pd.DataFrame(ph_rows, columns=["policy_id", "period", "premium_due", "premium_paid", "on_time_flag"])

# ---------------------------------------------------------------------------
# 5. MULTIPLY ENGAGEMENT / ACTIVE DAYZ / TIER HISTORY / REWARD REDEMPTIONS
# ---------------------------------------------------------------------------
unique_clients = sorted(set(client_id))
TIERS = ["Blue", "Bronze", "Silver", "Gold", "Diamond"]

eng_rows, dayz_rows = [], []
client_tier = {c: rng.choice(TIERS, p=[0.30, 0.28, 0.22, 0.14, 0.06]) for c in unique_clients}
for c in unique_clients:
    base_score = rng.uniform(35, 95)
    tier = client_tier[c]
    cmarket = rng.choice(MARKETS, p=MARKET_WEIGHTS)
    for period in hist_months:
        score = float(np.clip(base_score + rng.normal(0, 6), 5, 100))
        eng_rows.append((c, cmarket, tier, period, round(score, 1)))
        active_dayz = int(np.clip(score / 100 * 21 + rng.normal(0, 2), 0, 21))
        recharge_dayz = int(np.clip(active_dayz * rng.uniform(0.5, 0.9), 0, 14))
        dayz_rows.append((c, period, active_dayz, recharge_dayz))

multiply_engagement = pd.DataFrame(eng_rows, columns=["client_id", "market", "tier", "month", "engagement_score"])
active_dayz_df = pd.DataFrame(dayz_rows, columns=["client_id", "month", "active_dayz", "recharge_dayz"])

n_tier_changes = int(len(unique_clients) * 0.18)
tc_clients = rng.choice(unique_clients, size=n_tier_changes, replace=False)
tier_rows = []
for c in tc_clients:
    cur = client_tier[c]
    cur_i = TIERS.index(cur)
    direction = rng.choice(["upgrade", "downgrade"], p=[0.55, 0.45])
    new_i = min(cur_i + 1, len(TIERS) - 1) if direction == "upgrade" else max(cur_i - 1, 0)
    change_date = random_dates(START, TODAY, 1)[0]
    tier_rows.append((c, cur, TIERS[new_i], change_date, direction))
tier_history = pd.DataFrame(tier_rows, columns=["client_id", "from_tier", "to_tier", "change_date", "direction"])

n_redemptions = int(len(unique_clients) * 0.9)
partners = ["Woolworths", "British Airways", "Virgin Active", "Takealot", "Engen", "HBO Max"]
red_clients = rng.choice(unique_clients, size=n_redemptions, replace=True)
reward_redemptions = pd.DataFrame({
    "client_id": red_clients,
    "redemption_date": random_dates(START, TODAY, n_redemptions),
    "partner": rng.choice(partners, size=n_redemptions),
    "reward_value": rng.choice([50, 100, 150, 250, 400, 600], size=n_redemptions),
})

# ---------------------------------------------------------------------------
# 6. CLAIMS MASTER / CLAIMS BY CELL / FRAUD FLAGS / TRIAGE OUTCOMES
# ---------------------------------------------------------------------------
N_CLAIMS = 5200
claim_id = [f"CLM-{i:06d}" for i in range(1, N_CLAIMS + 1)]
claim_market = pick(MARKETS, MARKET_WEIGHTS, N_CLAIMS)
claim_product = pick(PRODUCTS, PRODUCT_WEIGHTS, N_CLAIMS)
claim_cell = rng.choice(CELLS, size=N_CLAIMS)
claim_policy_ref = rng.choice(policy_id, size=N_CLAIMS)
loss_date = random_dates(START, TODAY, N_CLAIMS)
submission_lag = rng.integers(0, 30, N_CLAIMS)
submission_date = pd.Series(loss_date + pd.to_timedelta(submission_lag, unit="D")).clip(upper=TODAY)
claim_amount_base = {"Funeral": 18000, "Life": 120000, "Health": 22000, "Investment": 60000}
claim_amount = np.array([
    max(500, rng.normal(claim_amount_base[p], claim_amount_base[p] * 0.45))
    for p in claim_product
]).round(2)

n_brokers, n_assessors, n_repairers = 60, 45, 50
broker_id = [f"BRK-{rng.integers(1, n_brokers + 1):03d}" for _ in range(N_CLAIMS)]
assessor_id = [f"ASR-{rng.integers(1, n_assessors + 1):03d}" for _ in range(N_CLAIMS)]
repairer_id = [f"RPR-{rng.integers(1, n_repairers + 1):03d}" for _ in range(N_CLAIMS)]

claims_master = pd.DataFrame({
    "claim_id": claim_id,
    "policy_id": claim_policy_ref,
    "market": claim_market,
    "product_line": claim_product,
    "cell": claim_cell,
    "loss_date": loss_date,
    "submission_date": submission_date,
    "claim_amount": claim_amount,
    "broker_id": broker_id,
    "assessor_id": assessor_id,
    "repairer_id": repairer_id,
})

# Fraud risk score, biased per cell story
fraud_base = rng.normal(28, 14, N_CLAIMS)
for i, cell in enumerate(claim_cell):
    fraud_base[i] *= FRAUD_BIAS_CELL.get(cell, 1.0)
fraud_risk_score = np.clip(fraud_base, 1, 100).round(1)


def tier_of(score):
    if score <= 30:
        return "Green"
    elif score <= 70:
        return "Amber"
    return "Red"


final_tier = [tier_of(s) for s in fraud_risk_score]
layer1a = [tier_of(np.clip(s + rng.normal(0, 5), 1, 100)) for s in fraud_risk_score]
layer1b = rng.choice(["Clear", "Suspected", "Confirmed"], size=N_CLAIMS, p=[0.90, 0.08, 0.02])

fraud_flags = pd.DataFrame({
    "claim_id": claim_id,
    "fraud_risk_score": fraud_risk_score,
    "layer1a_verdict": layer1a,
    "layer1b_verdict": layer1b,
    "final_tier": final_tier,
    "flagged_date": submission_date,
})

outcome_map = {"Green": "Auto-approved", "Amber": "Human review", "Red": "SIU referral"}
turnaround = []
for t in final_tier:
    if t == "Green":
        turnaround.append(round(rng.uniform(0.5, 4), 1))
    elif t == "Amber":
        turnaround.append(round(rng.uniform(24, 96), 1))
    else:
        turnaround.append(round(rng.uniform(96, 480), 1))

triage_outcomes = pd.DataFrame({
    "claim_id": claim_id,
    "tier": final_tier,
    "outcome": [outcome_map[t] for t in final_tier],
    "turnaround_hours": turnaround,
})

cb_rows = []
for cell in CELLS:
    for mkt in MARKETS:
        for period in MONTHS:
            mask = (
                (np.array(claim_cell) == cell)
                & (np.array(claim_market) == mkt)
                & (pd.Series(loss_date).dt.to_period("M") == period.to_period("M")).to_numpy()
            )
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            amt = float(claim_amount[mask].sum())
            cb_rows.append((cell, mkt, period, cnt, round(amt, 2), round(amt / cnt, 2)))
claims_by_cell = pd.DataFrame(cb_rows, columns=["cell", "market", "month", "claims_count", "total_claim_amount", "avg_claim_amount"])

# ---------------------------------------------------------------------------
# 7. ADVISOR PERFORMANCE / LEAD PIPELINE / CONVERSION RATES / TERRITORY DATA
# ---------------------------------------------------------------------------
advisor_ids = [f"ADV-{i:03d}" for i in range(1, n_advisors + 1)]
advisor_market = {a: rng.choice(MARKETS, p=MARKET_WEIGHTS) for a in advisor_ids}
advisor_territory = {a: rng.choice(TERRITORIES[advisor_market[a]]) for a in advisor_ids}

ap_rows = []
for a in advisor_ids:
    base_contact = rng.uniform(0.55, 0.92)
    base_conv = rng.uniform(0.12, 0.34)
    declining = advisor_territory[a] == DECLINING_TERRITORY
    for i, period in enumerate(hist_months):
        drift = -0.018 * i if declining else rng.normal(0, 0.01)
        contact_rate = float(np.clip(base_contact + drift + rng.normal(0, 0.03), 0.2, 0.99))
        conv_rate = float(np.clip(base_conv + drift + rng.normal(0, 0.02), 0.03, 0.55))
        leads_assigned = int(rng.integers(15, 80))
        leads_contacted = int(leads_assigned * contact_rate)
        conversions = int(leads_contacted * conv_rate)
        hours_to_contact = float(np.clip(rng.normal(30 if not declining else 55, 18), 1, 168))
        ap_rows.append((a, advisor_market[a], advisor_territory[a], period, leads_assigned,
                         leads_contacted, round(contact_rate, 3), conversions, round(conv_rate, 3),
                         round(hours_to_contact, 1)))

advisor_performance = pd.DataFrame(ap_rows, columns=[
    "advisor_id", "market", "territory", "month", "leads_assigned", "leads_contacted",
    "contact_rate", "conversions", "conversion_rate", "avg_hours_to_first_contact",
])

conversion_rates = (
    advisor_performance.assign(product_type=lambda d: pick(PRODUCTS, PRODUCT_WEIGHTS, len(d)))
    [["advisor_id", "market", "product_type", "month", "conversion_rate"]]
)

N_LEADS = 12000
stages = ["New", "Contacted", "Meeting", "Proposal", "Closed Won", "Closed Lost"]
stage_weights = [0.18, 0.27, 0.20, 0.14, 0.13, 0.08]
lead_advisor = rng.choice(advisor_ids, size=N_LEADS)
created_date = random_dates(START, TODAY, N_LEADS)
last_update_raw = created_date + pd.to_timedelta(rng.integers(0, 45, N_LEADS), unit="D")
last_update = pd.Series(last_update_raw).clip(upper=TODAY)
lead_pipeline = pd.DataFrame({
    "lead_id": [f"LEAD-{i:06d}" for i in range(1, N_LEADS + 1)],
    "advisor_id": lead_advisor,
    "market": [advisor_market[a] for a in lead_advisor],
    "territory": [advisor_territory[a] for a in lead_advisor],
    "product_interest": pick(PRODUCTS, PRODUCT_WEIGHTS, N_LEADS),
    "stage": pick(stages, stage_weights, N_LEADS),
    "created_date": created_date,
    "last_update": last_update,
})

td_rows = []
for mkt, territs in TERRITORIES.items():
    for t in territs:
        advs = [a for a in advisor_ids if advisor_territory[a] == t]
        leads_t = lead_pipeline[lead_pipeline["territory"] == t]
        td_rows.append((
            t, mkt, len(advs), len(leads_t),
            int((leads_t["stage"] == "Closed Won").sum()),
        ))
territory_data = pd.DataFrame(td_rows, columns=["territory", "market", "advisor_count", "total_leads", "total_conversions"])

# ---------------------------------------------------------------------------
# Write everything out
# ---------------------------------------------------------------------------
TABLES = {
    "claims_master": claims_master,
    "claims_by_cell": claims_by_cell,
    "fraud_flags": fraud_flags,
    "triage_outcomes": triage_outcomes,
    "policy_master": policy_master,
    "lapse_events": lapse_events,
    "new_business": new_business,
    "premium_history": premium_history,
    "multiply_engagement": multiply_engagement,
    "active_dayz": active_dayz_df,
    "tier_history": tier_history,
    "reward_redemptions": reward_redemptions,
    "advisor_performance": advisor_performance,
    "lead_pipeline": lead_pipeline,
    "conversion_rates": conversion_rates,
    "territory_data": territory_data,
}

if __name__ == "__main__":
    for name, df in TABLES.items():
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:<22} {len(df):>7,} rows  -> {path.name}")

    import json
    meta = {"as_of_date": TODAY.date().isoformat(), "window_start": START.date().isoformat()}
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Data window: {START.date()} -> {TODAY.date()}")
