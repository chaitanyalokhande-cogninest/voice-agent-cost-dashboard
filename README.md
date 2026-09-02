# Voice-call cost dashboard

This project estimates AI service costs from measured call duration and
transcript-derived usage. It is a budgeting estimate, not an invoice
reconciliation.

## Run locally

```powershell
.\.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

The dashboard can load a usage CSV from the project folder or accept one with
the upload control. The repository intentionally excludes recordings,
transcripts, generated CSVs, and `.env` because they may contain private call
data or credentials.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, choose **New app**.
3. Select the repository, branch, and `streamlit_app.py` as the main file.
4. Deploy the app.
5. Upload the usage CSV through the dashboard, or provide a sanitized dataset
   separately.

The dashboard itself does not need AWS credentials. Never commit `.env`, AWS
keys, recordings, transcripts, or raw customer data. Use `.env.example` as the
configuration template for local analyzer runs.
