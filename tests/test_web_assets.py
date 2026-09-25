import re
from html.parser import HTMLParser

from app.main import WEB_DIR


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value is not None:
                self.ids.append(value)


def test_dashboard_has_unique_ids_and_all_javascript_targets_exist():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    parser = IdCollector()
    parser.feed(html)

    assert len(parser.ids) == len(set(parser.ids))
    referenced_ids = set(re.findall(r'byId\("([^"]+)"\)', javascript))
    assert referenced_ids <= set(parser.ids)


def test_dashboard_assets_do_not_embed_backend_secrets():
    combined = "\n".join(
        (WEB_DIR / filename).read_text(encoding="utf-8")
        for filename in ("index.html", "styles.css", "app.js")
    )

    assert "SUPABASE_SERVICE_KEY" not in combined
    assert "private_key" not in combined
