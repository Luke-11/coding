from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

@dataclass
class TocEntry:
    level: str
    number: Optional[str]
    title: str
    page: Optional[str]
    secid: Optional[int] = None
    file: Optional[str] = None
    line: Optional[int] = None

def parse_toc_line(line: str) -> Optional[Tuple[str, str, str]]:
    r"""Parse a single \contentsline entry, handling nested braces in body."""
    # Match \contentsline {level}{body}{page}{label}
    # We need to find the boundaries by counting braces
    if not line.strip().startswith('\\contentsline'):
        return None
    
    # Find the opening brace after \contentsline
    start = line.find('{')
    if start == -1:
        return None
    
    # Find level (first argument)
    level_end = find_brace_end(line, start + 1)
    if level_end == -1:
        return None
    level = line[start + 1:level_end]
    
    # Find body (second argument) - starts after level's closing brace
    body_start = line.find('{', level_end + 1)
    if body_start == -1:
        return None
    body_end = find_brace_end(line, body_start + 1)
    if body_end == -1:
        return None
    body = line[body_start + 1:body_end]
    
    # Find page (third argument)
    page_start = line.find('{', body_end + 1)
    if page_start == -1:
        return None
    page_end = find_brace_end(line, page_start + 1)
    if page_end == -1:
        return None
    page = line[page_start + 1:page_end]
    
    return (level, body, page)

def find_brace_end(text: str, start: int) -> int:
    """Find the matching closing brace, handling nested braces."""
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1

NUMBERLINE_RE = re.compile(r"""\\numberline\s*\{(?P<num>[^}]+)\}(?P<title>.*)""", re.DOTALL)
SECID_RE = re.compile(r"""\[secid=(?P<secid>\d+)\]""")

AUX_NEWLABEL_RE = re.compile(
    r"""\\newlabel\{(?P<label>[^}]+)\}\{\{(?P<num>[^}]*)\}\{(?P<page>[^}]*)\}""",
    re.DOTALL,
)

# Matches lines in secid file:
# <secid>|<file>|<line>
SECTPOS_RE = re.compile(r"""^(?P<secid>\d+)\|(?P<file>[^|]+)\|(?P<line>\d+)\s*$""")

def tex_unbrace(s: str) -> str:
    return s.strip()

def parse_toc(toc_path: Path) -> List[TocEntry]:
    text = toc_path.read_text(encoding="utf-8", errors="replace")
    entries: List[TocEntry] = []
    for line in text.splitlines():
        result = parse_toc_line(line)
        if result is None:
            continue
        level, body, page = result
        
        level = level.strip()
        body = body.strip()
        page = page.strip()

        # Extract secid if present
        secid = None
        secid_match = SECID_RE.search(body)
        if secid_match:
            secid = int(secid_match.group("secid"))
            # Remove secid marker from body
            body = SECID_RE.sub("", body).strip()

        num = None
        title = body
        mn = NUMBERLINE_RE.match(body)
        if mn:
            num = mn.group("num").strip()
            title = mn.group("title").strip()

        # very light cleanup
        title = tex_unbrace(title)

        entries.append(TocEntry(level=level, number=num, title=title, page=page, secid=secid))
    return entries

def parse_aux_labels(aux_path: Path) -> Dict[str, Tuple[str, str]]:
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Tuple[str, str]] = {}
    for m in AUX_NEWLABEL_RE.finditer(text):
        out[m.group("label")] = (m.group("num"), m.group("page"))
    return out

def parse_sectpos(sectpos_path: Path) -> Dict[int, Tuple[str, int]]:
    # secid -> (file, line)
    out: Dict[int, Tuple[str, int]] = {}
    if not sectpos_path.exists():
        return out
    for line in sectpos_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SECTPOS_RE.match(line)
        if not m:
            continue
        secid = int(m.group("secid"))
        out[secid] = (m.group("file"), int(m.group("line")))
    return out

def attach_positions(entries: List[TocEntry], posmap: Dict[int, Tuple[str, int]]) -> None:
    """
    Match entries by their secid number to the position map.
    """
    for e in entries:
        if e.secid is not None and e.secid in posmap:
            e.file, e.line = posmap[e.secid]

if __name__ == "__main__":
    base = Path("snippets")
    toc = parse_toc(base / "test.toc")
    pos = parse_sectpos(base / "test.secid")
    attach_positions(toc, pos)

    for e in toc:
        loc = f"{Path(e.file).name}:{e.line}" if e.file else "?:?"
        num = e.number if e.number else "*"
        print(f"{num:>6}  {e.level:<10}  p.{e.page:<4}  {loc:<15}  {e.title}")
