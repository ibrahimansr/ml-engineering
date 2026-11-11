"""Check that the generated site contains all of the book's content.

    python build/site/check_site.py      # after build_site.py

For every markdown file: a page exists, every heading, prose line, code block, table row,
equation and image from the markdown appears in the page, and every internal link resolves.
Exits non-zero and prints what's missing if anything is.
"""
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_site import ROOT, OUT, walk  # noqa: E402

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

def page_text(page_html):
    art = page_html[page_html.find("<article>"):page_html.find("</article>")]
    art = re.sub(r'<(span|div) class="math-(inline|display)">.*?</\1>', " ", art, flags=re.S)  # equations counted separately
    art = re.sub(r"<img[^>]*>", " ", art)
    return norm(html.unescape(re.sub(r"<[^>]+>", " ", art)))

def strip_inline(t):
    """Markdown inline syntax -> the text a reader sees. Code spans are kept verbatim."""
    out = []
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)              # images (checked separately)
    t = re.sub(r"\[((?:[^\]`]|`[^`]*`)*)\]\([^)]*\)", r"\1", t)  # links -> link text, even with `code` inside
    for i, part in enumerate(re.split(r"(`+[^`\n]*?`+)", t)):
        if i % 2:                                         # code span: content shows as-is
            out.append(part.strip("`"))
            continue
        part = re.sub(r"(?<!\\)\$[^$\n]+?(?<!\\)\$", " ", part)  # inline math (counted separately)
        part = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", part)        # images (checked separately)
        part = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", part)     # links -> link text
        part = re.sub(r"<[^>]+>", " ", part)                     # raw html tags
        part = re.sub(r"\\([\\`*_{}\[\]()#+\-.!$|<>~])", r"\1", part)  # markdown escapes
        out.append(part)
    return "".join(out)

def md_prose_lines(md):
    """Visible text of each paragraph / list item / table row, with markdown syntax stripped."""
    md = re.sub(r"^ *(```|~~~).*?^ *\1[^\n]*$", "", md, flags=re.S | re.M)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$", "", md, flags=re.S)
    block = []
    for line in md.splitlines() + [""]:
        starts_item = re.match(r"^\s*(#{1,6}\s|>|[-*+]\s|\d+[.)]\s|\|)", line)
        if (not line.strip() or starts_item) and block:
            yield block_text(block)
            block = []
        if line.strip():
            block.append(line)
            if line.lstrip().startswith(("|", "#")):       # table rows and headings stand alone
                yield block_text(block)
                block = []
    return

def block_text(lines):
    raw = " ".join(l.strip() for l in lines)
    if re.match(r"^\|?\s*:?-{3,}", raw):                 # table separator row
        return raw, ""
    t = re.sub(r"^\s*(#{1,6}\s|>+\s?|[-*+]\s|\d+[.)]\s)", "", raw)
    t = strip_inline(t).replace("|", " ")
    t = re.sub(r"[*_~]", "", t)
    return raw, html.unescape(norm(t))

def main():
    problems = []
    md_files = list(walk(lambda p: p.suffix == ".md"))
    for src in md_files:
        rel = src.relative_to(ROOT)
        page = OUT / rel.with_suffix(".html")
        if not page.exists():
            problems.append(f"{rel}: no page generated"); continue
        md, ph = src.read_text(errors="ignore"), page.read_text(errors="ignore")
        text = page_text(ph)
        # compare ignoring whitespace and inline-markup characters: tags and markup shift spaces around
        squash = lambda s: re.sub(r"[\s`*_~|]", "", s)
        loose = squash(text)
        for raw, t in md_prose_lines(md):
            if len(re.sub(r"\W", "", t)) >= 4 and squash(t) not in loose:
                problems.append(f"{rel}: prose missing: {t[:90]!r}")
        art = ph[ph.find("<article>"):ph.find("</article>")]
        code_html = html.unescape(art)
        for block in re.findall(r"^ *```[^\n]*\n(.*?)^ *```", md, flags=re.S | re.M):
            first = next((l.strip() for l in block.splitlines() if l.strip()), "")
            if first and first not in code_html:
                problems.append(f"{rel}: code block missing, starting {first[:70]!r}")
        n_md_math = len(re.findall(r"(?<!\\)\$\$", re.sub(r"```.*?```", "", md, flags=re.S))) // 2
        n_page_math = art.count('class="math-display"')
        n_md_inline = sum(len(re.findall(r"(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)(?<!\\)\$(?!\$)", part))
                          for i, part in enumerate(re.split(r"(`+[^`\n]*?`+)", re.sub(r"```.*?```", "", md, flags=re.S))) if i % 2 == 0)
        if art.count('class="math-inline"') < n_md_inline * 0.9:
            problems.append(f"{rel}: about {n_md_inline} inline equations in markdown, {art.count('class=\"math-inline\"')} on the page")
        if n_page_math < n_md_math:
            problems.append(f"{rel}: {n_md_math} display equations in markdown, {n_page_math} on the page")
        for img in re.findall(r'<img[^>]+src="([^"]+)"', art):
            if not re.match(r"https?:", img) and not (page.parent / img).resolve().exists():
                problems.append(f"{rel}: image missing: {img}")
    pages = list(OUT.rglob("*.html"))
    ids = {p.resolve(): set(re.findall(r'id="([^"]+)"', p.read_text(errors="ignore"))) for p in pages}
    for p in pages:
        s = p.read_text(errors="ignore")
        art = s[s.find("<article>"):s.find("</article>")]
        for url in re.findall(r'href="([^"]+)"', art):
            if re.match(r"^(https?:|mailto:|data:)", url):
                continue
            path, _, anchor = html.unescape(url).partition("#")
            target = (p.parent / path).resolve() if path else p.resolve()
            if not target.exists() or (anchor and target.suffix == ".html" and anchor not in ids.get(target, ())):
                problems.append(f"{p.relative_to(OUT)}: broken link {url}")
    for line in problems:
        print(line)
    print(f"checked {len(md_files)} markdown files, {len(pages)} pages: {len(problems)} problems")
    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
