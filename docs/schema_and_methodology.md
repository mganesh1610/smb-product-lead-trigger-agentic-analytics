# Schema Contract, Statistical Formulations & Regulated KPI Governance

## 1. Public Datasets & SHA-256 Provenance (`58,294` Combined Records)

| Dataset | Source URL | Verified Scale | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **UCI Bank Marketing Financial Lead & Trigger Dataset** | `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` | `45,211` rows × `17` fields | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |
| **PolyAI Banking77 Servicing Corpus (Train)** | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv` | `10,003` rows × `2` fields | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| **PolyAI Banking77 Servicing Corpus (Holdout Test)** | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv` | `3,080` rows × `2` fields | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

## 2. Relational Table Schemas (`SQLite` / `Snowflake` / `BigQuery` Compatible)

1. **`dim_smb_clients` (`45,211` Small Business Client Profiles):**
   - `client_id` (`TEXT PRIMARY KEY`), `smb_segment` (`Micro-Merchant & Sole Proprietor`, `Professional & Technical Services`, `Retail & Hospitality SMB`, `Growth Commercial Enterprise`), `job_role_raw` (`TEXT`), `annual_liquidity_balance` (`INTEGER`), `has_commercial_credit` (`INTEGER`), `has_working_capital_loan` (`INTEGER`), `credit_default_flag` (`INTEGER`).
2. **`fact_lead_trigger_events` (`45,211` Lead & Trigger Journey Events):**
   - `event_id` (`TEXT PRIMARY KEY`), `client_id` (`TEXT FOREIGN KEY`), `trigger_type` (`Liquidity Surge & Treasury Sweep Trigger`, `Working Capital & Credit Line Trigger`, `Lifecycle Re-Engagement & Dormancy Trigger`, `Seasonal Merchant Cash-Flow Trigger`, `Digital Payment Acceptance Onboarding Trigger`), `discovery_channel` (`TEXT`), `experiment_arm` (`Treatment_Trigger`, `Control_Static`), `contact_duration_sec` (`INTEGER`), `campaign_touch_count` (`INTEGER`), `prior_days_since_touch` (`INTEGER`), `prior_touch_outcomes` (`TEXT`), `propensity_score` (`REAL`), `converted_flag` (`INTEGER`), `incremental_revenue_usd` (`REAL`), `kyc_control_passed` (`INTEGER`), `audit_log_complete` (`INTEGER`).
3. **`fact_servicing_text_inquiries` (`13,083` Regulated Servicing & Discovery Inquiries):**
   - `inquiry_id` (`TEXT PRIMARY KEY`), `smb_segment` (`TEXT`), `raw_intent_category` (`77` fine-grained banking intents), `servicing_pathway` (`Card & Payment Acceptance`, `Working Capital & Liquidity Management`, `ACH, Wire & Real-Time Settlement`, `Digital Onboarding & KYC Controls`, `Fee Transparency & Dispute Resolution`, `Account Security & Fraud Controls`), `utterance_text` (`TEXT`), `ai_resolved_flag` (`INTEGER`), `resolution_latency_sec` (`REAL`).

## 3. Mathematical & Statistical Formulations

- **L2-Regularized Logistic Lead-Trigger Propensity Model:**
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] + \frac{\lambda}{2N}\sum_{j=1}^{d} w_j^2, \quad \hat{p}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$$
- **Doubly Robust Augmented Inverse Propensity Weighting (AIPW) ATE:**
  $$\hat{\tau}_{\text{AIPW}} = \frac{1}{N}\sum_{i=1}^{N}\left[\left(\hat{\mu}_1(\mathbf{x}_i) - \hat{\mu}_0(\mathbf{x}_i)\right) + \frac{T_i (Y_i - \hat{\mu}_1(\mathbf{x}_i))}{\hat{e}(\mathbf{x}_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(\mathbf{x}_i))}{1-\hat{e}(\mathbf{x}_i)}\right]$$
- **Sublinear TF-IDF & Complement Log-Odds Servicing Pathway Classifier:**
  $$\text{tfidf}(t, d) = \left(1 + \log \text{tf}(t, d)\right)\cdot \left(\log\frac{1 + N_{\text{train}}}{1 + \text{df}(t)} + 1\right), \quad w_{c, t} = \log\left(\frac{\theta_{c,t}}{\theta_{\bar{c},t}}\right)$$
