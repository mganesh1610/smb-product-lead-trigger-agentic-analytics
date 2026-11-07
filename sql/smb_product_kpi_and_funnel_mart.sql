-- =============================================================================
-- Small Business Product Analytics, Lead-Trigger Funnel & Regulated Day-One KPI Mart
-- Compatible with Snowflake, BigQuery, DuckDB, PostgreSQL, and SQLite 3.35+
-- Uses SQL Joins & CTEs and Window Functions across 5 Regulated Dimensions:
-- Financial, Operational, Controls, Audit, and Product
-- =============================================================================

WITH client_segment_base AS (
    SELECT
        c.client_id,
        c.smb_segment,
        c.job_role_raw,
        c.annual_liquidity_balance,
        c.has_commercial_credit,
        c.has_working_capital_loan,
        c.credit_default_flag,
        e.event_id,
        e.trigger_type,
        e.discovery_channel,
        e.experiment_arm,
        e.contact_duration_sec,
        e.campaign_touch_count,
        e.prior_days_since_touch,
        e.prior_touch_outcomes,
        e.propensity_score,
        e.converted_flag,
        e.incremental_revenue_usd,
        e.kyc_control_passed,
        e.audit_log_complete
    FROM dim_smb_clients AS c
    INNER JOIN fact_lead_trigger_events AS e
        ON c.client_id = e.client_id
),
ranked_trigger_deciles AS (
    SELECT
        csb.*,
        NTILE(10) OVER (ORDER BY csb.propensity_score DESC) AS propensity_decile,
        ROW_NUMBER() OVER (
            PARTITION BY csb.smb_segment, csb.trigger_type
            ORDER BY csb.propensity_score DESC, csb.client_id ASC
        ) AS segment_trigger_rank
    FROM client_segment_base AS csb
),
servicing_pathway_Rollup AS (
    SELECT
        smb_segment,
        servicing_pathway,
        COUNT(*) AS inquiry_volume,
        AVG(CASE WHEN ai_resolved_flag = 1 THEN 1.0 ELSE 0.0 END) AS ai_deflection_rate,
        AVG(resolution_latency_sec) AS avg_resolution_latency_sec
    FROM fact_servicing_text_inquiries
    GROUP BY smb_segment, servicing_pathway
),
day_one_regulated_kpis AS (
    SELECT
        r.smb_segment,
        r.trigger_type,
        r.experiment_arm,
        COUNT(*) AS total_targeted_clients,
        SUM(r.converted_flag) AS total_conversions,
        ROUND(AVG(CAST(r.converted_flag AS REAL)) * 100.0, 4) AS conversion_rate_pct,
        ROUND(SUM(r.incremental_revenue_usd), 2) AS total_incremental_revenue_usd,
        ROUND(AVG(r.incremental_revenue_usd), 2) AS avg_revenue_per_client_usd,
        ROUND(AVG(r.propensity_score), 4) AS mean_propensity_score,
        ROUND(AVG(CASE WHEN r.propensity_decile <= 2 THEN CAST(r.converted_flag AS REAL) END) * 100.0, 4) AS top_20pct_conversion_rate_pct,
        ROUND(AVG(CAST(r.kyc_control_passed AS REAL)) * 100.0, 4) AS controls_kyc_pass_rate_pct,
        ROUND(AVG(CAST(r.audit_log_complete AS REAL)) * 100.0, 4) AS audit_traceability_pct
    FROM ranked_trigger_deciles AS r
    GROUP BY r.smb_segment, r.trigger_type, r.experiment_arm
)
SELECT
    k.smb_segment,
    k.trigger_type,
    k.experiment_arm,
    k.total_targeted_clients,
    k.total_conversions,
    k.conversion_rate_pct,
    k.total_incremental_revenue_usd,
    k.avg_revenue_per_client_usd,
    k.mean_propensity_score,
    k.top_20pct_conversion_rate_pct,
    k.controls_kyc_pass_rate_pct,
    k.audit_traceability_pct
FROM day_one_regulated_kpis AS k
ORDER BY k.smb_segment ASC, k.trigger_type ASC, k.experiment_arm ASC;
