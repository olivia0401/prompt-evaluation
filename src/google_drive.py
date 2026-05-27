"""
Shared Google Drive / Docs / Sheets helpers.

ONE-TIME SETUP:
  1. Enable BOTH Google Drive API AND Google Sheets API in Google Cloud
     Console. Drive API alone is NOT enough — upload_xlsx_to_replace_sheets
     does an xlsx→Sheets conversion internally, and Drive silently fails
     with 403 "insufficientFilePermissions" if Sheets API is disabled.
       - https://console.cloud.google.com/apis/library/drive.googleapis.com
       - https://console.cloud.google.com/apis/library/sheets.googleapis.com
  2. Create OAuth Client ID (Desktop app) → download credentials.json
  3. Place credentials.json at project root
  4. First run opens browser; token.json is cached for reuse

SCOPE NOTE:
We use the full `drive` scope (not the narrower `drive.file`) because
build_xlsx needs to UPDATE an existing user-owned Google Sheet — not just
files our app created. If you previously authorized with `drive.file`,
delete token.json so the next run re-authorizes with the broader scope.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from src import config as cfg


# Full drive scope — required to update arbitrary user-owned files by ID.
# drive.file only grants access to files created/opened by THIS app, which
# is not enough to replace an existing Sheets file the user already owns.
SCOPES = ["https://www.googleapis.com/auth/drive"]

CRED_PATH = cfg.PROJECT_ROOT / "credentials.json"
TOKEN_PATH = cfg.PROJECT_ROOT / "token.json"


# ---------- OAuth ----------

def get_credentials():
    """Return refreshed/cached credentials, or run the browser flow on first use."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise SystemExit(
            "Google API libraries not installed. Run:\n"
            "  pip install -r requirements.txt\n"
            f"(missing: {e})"
        )

    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except ValueError:
            # Stale token (e.g. authorized under a narrower scope before the upgrade).
            # Force a fresh run-through.
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if not CRED_PATH.exists():
                raise SystemExit(
                    f"Missing {CRED_PATH}. Setup steps:\n"
                    "  1. https://console.cloud.google.com/ → create/select project\n"
                    "  2. Enable 'Google Drive API'\n"
                    "  3. Credentials → Create → OAuth Client ID → Desktop app\n"
                    "  4. Download JSON, save to credentials.json in project root\n"
                    f"     Expected at: {CRED_PATH}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return creds


# ---------- Drive: HTML → new Google Doc ----------

def upload_html_as_google_doc(html_str: str, title: str,
                              folder_id: Optional[str] = None) -> str:
    """Upload HTML content to Drive, converting to a new Google Doc.
    Returns the new doc's webViewLink (URL). Each invocation creates a NEW Doc."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    service = build("drive", "v3", credentials=get_credentials(), cache_discovery=False)
    file_metadata = {
        "name": title,
        "mimeType": "application/vnd.google-apps.document",
    }
    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaIoBaseUpload(
        BytesIO(html_str.encode("utf-8")), mimetype="text/html", resumable=False
    )
    f = service.files().create(
        body=file_metadata, media_body=media, fields="id,webViewLink"
    ).execute()
    return f["webViewLink"]


# ---------- Drive: xlsx → REPLACE existing Google Sheets ----------

def upload_xlsx_to_replace_sheets(xlsx_path: Path, sheet_id: str) -> str:
    """
    Upload a local .xlsx file's content as the new content of an EXISTING
    Google Sheets file (identified by sheet_id). Drive performs the
    xlsx→Sheets conversion automatically, preserving:
      - cell values, fonts, fills, bold/italic
      - merged cells
      - conditional formatting (heatmap color scales)
      - embedded images (PNG)

    Caveats:
      - This REPLACES the target Sheet's contents entirely. Any manual tabs
        in the Sheet that are not in the xlsx will be removed.
      - The user authenticated via OAuth must have edit permission on the
        target Sheet.
      - Requires the full 'drive' scope (not drive.file) because the target
        Sheet was not created by our app.

    Returns the Sheet's webViewLink (URL, stable across updates).
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    service = build("drive", "v3", credentials=get_credentials(), cache_discovery=False)

    media = MediaFileUpload(
        str(xlsx_path),
        mimetype=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        resumable=False,
    )

    # files.update with xlsx content on a Google Sheets file → in-place
    # content replacement (file ID stays the same; URL stays the same).
    f = service.files().update(
        fileId=sheet_id,
        media_body=media,
        fields="id,webViewLink",
    ).execute()
    return f.get("webViewLink", f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit")
