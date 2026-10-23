Investment Agent v5.4 — Self-service accounts + portfolio import

What's new:
- Sign in, self-service account creation, and password reset flow.
- Explicit explanations for pending email confirmation and Supabase email rate limits.
- Import CSV/XLSX from your old portfolio, with column mapping and preview.
- Replace or merge existing holdings, with confirmation before saving.
- Download private portfolio as CSV.
- Keeps the v5.3 market scan, score percentage, company briefs and portfolio analytics.

INSTALL
1. In GitHub repository arielporat71/investment-agent replace app.py with this archive's app.py.
2. Keep requirements.txt with supabase>=2.10,<3. To be safe, replace requirements.txt with this archive's version too.
3. DO NOT overwrite any existing portfolio.csv (old portfolio) or Supabase database.
4. Existing Streamlit Secrets remain: SUPABASE_URL and SUPABASE_ANON_KEY (publishable key). Never add a service_role or secret API key.
5. In Supabase SQL Editor run supabase_setup.sql only if portfolios table and RLS policies were not previously set up.
6. To import old holdings, first sign in, select Update portfolio, upload portfolio.csv or an Excel file, preview and confirm.

IMPORTANT ABOUT SELF-SERVICE REGISTRATION
The app cannot override Supabase Auth email confirmation or email sending rate limits. For reliable public self-service registration, configure custom SMTP in Supabase -> Authentication -> SMTP Settings, keep email confirmation enabled, and test a new account end-to-end. An existing unconfirmed user still needs confirmation. Avoid disabling confirmation for an open public signup without other safeguards.

DATA SECURITY
Portfolio access uses Supabase authentication and row-level security policies keyed by auth.uid(). Each user can only read and write their own holdings when policies are installed. Do not use a service role key in this app.

IMPORT FORMAT
CSV or XLSX, headers e.g. ticker,shares,avg_buy_price. Examples: NVDA,10,120.50. Import overwrites matching tickers in Merge mode. Replace mode replaces your account's portfolio. Check preview before confirming. This is not an automatic brokerage sync.
