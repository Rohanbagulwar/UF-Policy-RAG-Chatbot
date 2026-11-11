from pathlib import Path
from urllib.parse import urlparse

from openpyxl import load_workbook

from ..config import PROJECT_ROOT


SOURCE_WORKBOOK = (
    PROJECT_ROOT / "data" / "raw" / "Web_Scrapped_Dataset" / "policies_data.xlsx"
)

SOURCE_OVERRIDES = {
    "remote work location": (
        "Policy on Limited Remote Work",
        "https://policy.ufl.edu/policy/limited-remote-work/",
    ),
}


class PolicySourceCatalog:
    """Resolve indexed policy titles to their original source webpages."""

    def __init__(self, workbook_path: Path = SOURCE_WORKBOOK):
        if not workbook_path.is_file():
            raise FileNotFoundError(f"Policy source workbook not found: {workbook_path}")

        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
        try:
            if "Policies" not in workbook.sheetnames:
                raise ValueError(f"Policies sheet not found in {workbook_path}")

            worksheet = workbook["Policies"]
            rows = worksheet.iter_rows(values_only=True)
            headers = next(rows, None)
            if headers is None:
                raise ValueError(f"Policy source workbook is empty: {workbook_path}")

            columns = {str(value).strip().lower(): index for index, value in enumerate(headers) if value}
            if "title" not in columns or "link" not in columns:
                raise ValueError(f"Policy source workbook must contain title and link columns: {workbook_path}")

            self._sources_by_title: dict[str, tuple[str, str]] = {}
            for row in rows:
                title = row[columns["title"]]
                link = row[columns["link"]]
                if not title or not link:
                    continue

                normalized_title = str(title).strip().casefold()
                url = str(link).strip()
                parsed_url = urlparse(url)
                if (
                    parsed_url.scheme == "https"
                    and parsed_url.hostname == "policy.ufl.edu"
                    and parsed_url.netloc
                ):
                    self._sources_by_title.setdefault(
                        normalized_title,
                        (str(title).strip(), url),
                    )
        finally:
            workbook.close()

        self._sources_by_title.update(SOURCE_OVERRIDES)

    def get_source(self, title: str) -> tuple[str, str] | None:
        """Return the current official title and URL for an indexed title."""
        return self._sources_by_title.get(title.strip().casefold())

    def get_url(self, title: str) -> str | None:
        """Return the HTTPS source URL for a policy title, when available."""
        source = self.get_source(title)
        return source[1] if source else None
