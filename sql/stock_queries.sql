-- ============================================================
-- Stock Market Risk Analytics — SQL Analysis Queries
-- Database: MySQL  |  Dataset: NIFTY 500 Historical Prices
-- Author:   Ashraf Bukhari
-- ============================================================


-- ── QUERY 1: Annual Return per Stock ──────────────────────────────────────
SELECT
    ticker,
    sector,
    YEAR(date)                                         AS year,
    ROUND(MIN(adj_close), 2)                           AS year_open,
    ROUND(MAX(adj_close), 2)                           AS year_high,
    ROUND(MAX(adj_close) - MIN(adj_close), 2)          AS year_range,
    ROUND(
        (LAST_VALUE(adj_close) OVER (
            PARTITION BY ticker, YEAR(date)
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
         ) -
         FIRST_VALUE(adj_close) OVER (
            PARTITION BY ticker, YEAR(date)
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
         )
        ) / FIRST_VALUE(adj_close) OVER (
            PARTITION BY ticker, YEAR(date)
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) * 100, 2)                                    AS annual_return_pct
FROM stock_prices
WHERE adj_close IS NOT NULL
GROUP BY ticker, sector, YEAR(date)
ORDER BY ticker, year;


-- ── QUERY 2: Rolling 20-Day Volatility per Stock ──────────────────────────
SELECT
    date,
    ticker,
    sector,
    adj_close,
    ROUND(daily_return * 100, 4)                       AS daily_return_pct,
    ROUND(
        STDDEV(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) * SQRT(252) * 100, 2)                        AS rolling_vol_20d_annualised
FROM (
    SELECT
        date, ticker, sector, adj_close,
        (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
        / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)
        AS daily_return
    FROM stock_prices
) sub
ORDER BY ticker, date;


-- ── QUERY 3: Sector-Level Risk & Return Summary ───────────────────────────
WITH daily_returns AS (
    SELECT
        date, ticker, sector,
        (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
        / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)  AS daily_return
    FROM stock_prices
)
SELECT
    sector,
    COUNT(DISTINCT ticker)                             AS stocks,
    ROUND(AVG(daily_return) * 252 * 100, 2)           AS avg_ann_return_pct,
    ROUND(STDDEV(daily_return) * SQRT(252) * 100, 2)  AS avg_ann_volatility_pct,
    ROUND(
        (AVG(daily_return) - (0.065/252))
        / STDDEV(daily_return) * SQRT(252), 3)         AS avg_sharpe_ratio,
    ROUND(
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END)
        / COUNT(*) * 100, 1)                           AS positive_days_pct
FROM daily_returns
WHERE daily_return IS NOT NULL
GROUP BY sector
ORDER BY avg_sharpe_ratio DESC;


-- ── QUERY 4: Maximum Drawdown per Stock ───────────────────────────────────
WITH price_peaks AS (
    SELECT
        date, ticker, sector, adj_close,
        MAX(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                              AS running_max
    FROM stock_prices
),
drawdowns AS (
    SELECT
        date, ticker, sector, adj_close, running_max,
        (adj_close - running_max) / running_max * 100  AS drawdown_pct
    FROM price_peaks
)
SELECT
    ticker, sector,
    ROUND(MIN(drawdown_pct), 2)                        AS max_drawdown_pct,
    MIN(date)                                          AS drawdown_date,
    COUNT(CASE WHEN drawdown_pct < -10 THEN 1 END)    AS days_below_10pct_drawdown,
    COUNT(CASE WHEN drawdown_pct < -20 THEN 1 END)    AS days_below_20pct_drawdown
FROM drawdowns
GROUP BY ticker, sector
ORDER BY max_drawdown_pct ASC;


-- ── QUERY 5: Beta Calculation vs Market Index ─────────────────────────────
WITH stock_returns AS (
    SELECT
        s.date, s.ticker, s.sector,
        (s.adj_close - LAG(s.adj_close) OVER (PARTITION BY s.ticker ORDER BY s.date))
        / LAG(s.adj_close) OVER (PARTITION BY s.ticker ORDER BY s.date)  AS stock_return,
        (m.index_close - LAG(m.index_close) OVER (ORDER BY m.date))
        / LAG(m.index_close) OVER (ORDER BY m.date)                      AS market_return
    FROM stock_prices s
    JOIN market_index m ON s.date = m.date
)
SELECT
    ticker, sector,
    ROUND(
        (SUM(stock_return * market_return) - COUNT(*) * AVG(stock_return) * AVG(market_return))
        /
        (SUM(market_return * market_return) - COUNT(*) * POW(AVG(market_return), 2))
    , 3)                                               AS beta,
    CASE
        WHEN (SUM(stock_return * market_return) - COUNT(*) * AVG(stock_return) * AVG(market_return))
             / (SUM(market_return * market_return) - COUNT(*) * POW(AVG(market_return), 2)) < 0.7
        THEN 'Defensive'
        WHEN (SUM(stock_return * market_return) - COUNT(*) * AVG(stock_return) * AVG(market_return))
             / (SUM(market_return * market_return) - COUNT(*) * POW(AVG(market_return), 2)) > 1.2
        THEN 'Aggressive'
        ELSE 'Neutral'
    END                                                AS stock_type
FROM stock_returns
WHERE stock_return IS NOT NULL AND market_return IS NOT NULL
GROUP BY ticker, sector
ORDER BY beta ASC;


-- ── QUERY 6: Stock Composite Score (Custom KPI) ───────────────────────────
WITH metrics AS (
    SELECT
        ticker, sector,
        AVG(daily_return) * 252                        AS ann_return,
        STDDEV(daily_return) * SQRT(252)               AS ann_vol,
        (AVG(daily_return) - (0.065/252))
        / STDDEV(daily_return) * SQRT(252)             AS sharpe_ratio
    FROM (
        SELECT ticker, sector,
               (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
               / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS daily_return
        FROM stock_prices
    ) r WHERE daily_return IS NOT NULL
    GROUP BY ticker, sector
)
SELECT
    ticker, sector,
    ROUND(ann_return * 100, 2)                         AS ann_return_pct,
    ROUND(ann_vol * 100, 2)                            AS ann_vol_pct,
    ROUND(sharpe_ratio, 3)                             AS sharpe_ratio,
    ROUND(
        LEAST(GREATEST(sharpe_ratio / 2.0, 0), 1) * 35    -- Sharpe score (35%)
      + LEAST(GREATEST(1 - ann_vol/0.5, 0), 1) * 30       -- Volatility score (30%)
      + LEAST(GREATEST(ann_return/0.3, 0), 1) * 20         -- Return score (20%)
      + LEAST(GREATEST(1 - ABS(sharpe_ratio-1)/2, 0),1)*15 -- Consistency (15%)
    * 100, 1)                                          AS composite_score,
    CASE
        WHEN ROUND(LEAST(GREATEST(sharpe_ratio/2,0),1)*35
                 + LEAST(GREATEST(1-ann_vol/0.5,0),1)*30
                 + LEAST(GREATEST(ann_return/0.3,0),1)*20
                 + LEAST(GREATEST(1-ABS(sharpe_ratio-1)/2,0),1)*15*100,1) >= 75 THEN 'Elite ⭐'
        WHEN ROUND(LEAST(GREATEST(sharpe_ratio/2,0),1)*35
                 + LEAST(GREATEST(1-ann_vol/0.5,0),1)*30
                 + LEAST(GREATEST(ann_return/0.3,0),1)*20
                 + LEAST(GREATEST(1-ABS(sharpe_ratio-1)/2,0),1)*15*100,1) >= 55 THEN 'Strong 💚'
        WHEN ROUND(LEAST(GREATEST(sharpe_ratio/2,0),1)*35
                 + LEAST(GREATEST(1-ann_vol/0.5,0),1)*30
                 + LEAST(GREATEST(ann_return/0.3,0),1)*20
                 + LEAST(GREATEST(1-ABS(sharpe_ratio-1)/2,0),1)*15*100,1) >= 35 THEN 'Average 🟡'
        ELSE 'Weak 🔴'
    END                                                AS tier
FROM metrics
ORDER BY composite_score DESC;


-- ── QUERY 7: Rolling Correlation Between Two Sectors ──────────────────────
WITH it_returns AS (
    SELECT date, AVG(daily_ret) AS it_ret
    FROM (
        SELECT date, ticker,
               (adj_close-LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
               /LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS daily_ret
        FROM stock_prices WHERE sector='IT'
    ) r WHERE daily_ret IS NOT NULL GROUP BY date
),
bank_returns AS (
    SELECT date, AVG(daily_ret) AS bank_ret
    FROM (
        SELECT date, ticker,
               (adj_close-LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
               /LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS daily_ret
        FROM stock_prices WHERE sector='Banking'
    ) r WHERE daily_ret IS NOT NULL GROUP BY date
)
SELECT
    i.date,
    ROUND(i.it_ret * 100, 4)   AS it_daily_return_pct,
    ROUND(b.bank_ret * 100, 4) AS banking_daily_return_pct,
    -- Rolling 60-day correlation (approximated via covariance)
    ROUND(
        (SUM(i.it_ret * b.bank_ret) OVER (ORDER BY i.date ROWS 59 PRECEDING)
         - COUNT(*) OVER (ORDER BY i.date ROWS 59 PRECEDING)
           * AVG(i.it_ret) OVER (ORDER BY i.date ROWS 59 PRECEDING)
           * AVG(b.bank_ret) OVER (ORDER BY i.date ROWS 59 PRECEDING))
        /
        NULLIF(STDDEV(i.it_ret) OVER (ORDER BY i.date ROWS 59 PRECEDING)
             * STDDEV(b.bank_ret) OVER (ORDER BY i.date ROWS 59 PRECEDING), 0)
    , 3)                       AS rolling_60d_correlation
FROM it_returns i
JOIN bank_returns b ON i.date = b.date
ORDER BY i.date;


-- ── QUERY 8: Top 10 Stocks by Sharpe for Each Year ────────────────────────
WITH yearly_sharpe AS (
    SELECT
        ticker, sector, YEAR(date) AS yr,
        (AVG(daily_ret) - (0.065/252))
        / NULLIF(STDDEV(daily_ret), 0) * SQRT(252)    AS sharpe
    FROM (
        SELECT date, ticker, sector,
               (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
               / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS daily_ret
        FROM stock_prices
    ) r WHERE daily_ret IS NOT NULL
    GROUP BY ticker, sector, YEAR(date)
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY yr ORDER BY sharpe DESC) AS rnk
    FROM yearly_sharpe
)
SELECT yr, ticker, sector, ROUND(sharpe, 3) AS sharpe_ratio, rnk AS yearly_rank
FROM ranked
WHERE rnk <= 5
ORDER BY yr, rnk;


-- ── QUERY 9: Portfolio Comparison — Defensive vs Aggressive ───────────────
WITH returns AS (
    SELECT date, ticker, sector,
           (adj_close - LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date))
           / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS daily_ret
    FROM stock_prices
),
portfolio_returns AS (
    SELECT date,
        AVG(CASE WHEN sector IN ('Pharma','FMCG') THEN daily_ret END)   AS defensive_ret,
        AVG(CASE WHEN sector IN ('Metals','Auto','IT') THEN daily_ret END) AS aggressive_ret
    FROM returns WHERE daily_ret IS NOT NULL
    GROUP BY date
)
SELECT
    'Defensive (Pharma+FMCG)'                         AS portfolio,
    ROUND(AVG(defensive_ret)*252*100, 2)               AS ann_return_pct,
    ROUND(STDDEV(defensive_ret)*SQRT(252)*100, 2)      AS ann_vol_pct,
    ROUND((AVG(defensive_ret)-(0.065/252))
          /STDDEV(defensive_ret)*SQRT(252), 3)         AS sharpe_ratio
FROM portfolio_returns
UNION ALL
SELECT
    'Aggressive (Metals+Auto+IT)',
    ROUND(AVG(aggressive_ret)*252*100, 2),
    ROUND(STDDEV(aggressive_ret)*SQRT(252)*100, 2),
    ROUND((AVG(aggressive_ret)-(0.065/252))
          /STDDEV(aggressive_ret)*SQRT(252), 3)
FROM portfolio_returns;


-- ── QUERY 10: 52-Week High/Low Tracker ────────────────────────────────────
WITH price_range AS (
    SELECT
        ticker, sector,
        MAX(CASE WHEN date >= DATE_SUB('2023-12-31', INTERVAL 52 WEEK)
                 THEN adj_close END)                   AS high_52w,
        MIN(CASE WHEN date >= DATE_SUB('2023-12-31', INTERVAL 52 WEEK)
                 THEN adj_close END)                   AS low_52w,
        MAX(CASE WHEN date = '2023-12-29' THEN adj_close END) AS current_price
    FROM stock_prices
    GROUP BY ticker, sector
)
SELECT
    ticker, sector,
    ROUND(current_price, 2)                            AS current_price,
    ROUND(high_52w, 2)                                 AS high_52w,
    ROUND(low_52w, 2)                                  AS low_52w,
    ROUND((current_price - low_52w) / low_52w * 100, 1) AS pct_above_52w_low,
    ROUND((high_52w - current_price) / high_52w * 100, 1) AS pct_below_52w_high,
    CASE
        WHEN current_price >= high_52w * 0.95 THEN '🟢 Near 52W High'
        WHEN current_price <= low_52w  * 1.05 THEN '🔴 Near 52W Low'
        ELSE '🟡 Mid Range'
    END                                                AS position
FROM price_range
WHERE current_price IS NOT NULL
ORDER BY pct_above_52w_low DESC;
