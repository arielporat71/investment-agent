# Investment Agent v3 - Stooq fallback

גרסה זו מפחיתה משמעותית את בעיית `YFRateLimitError` של Yahoo Finance.

מה חדש:
- שימוש ב-Stooq כעדיפות ראשונה לנתוני סוף-יום.
- Yahoo Finance רק כגיבוי.
- Cache מקומי בתיקיית `.cache`.

הרצה:
```bash
py -3.12 -m streamlit run app.py
```

אם הדשבורד כבר פתוח, עצור עם `Ctrl + C`, חלץ את התיקייה החדשה והריץ שוב.

הערה: הנתונים עשויים להיות delayed/end-of-day, וזה מתאים לסוכן המלצות יומי/חצי-יומי.
