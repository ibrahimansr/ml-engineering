"""Build a static HTML website from every markdown file in the repo.

    python build/site/build_site.py            # writes ./site
    cd site && python -m http.server 8000      # preview at http://localhost:8000

The output directory is plain static files: deploy it to Vercel (or any static host) as-is.
Needs: markdown-it-py, mdit-py-plugins (pip install -r build/requirements.txt).
Equations ($...$ and $$...$$, the same syntax GitHub renders) are typeset in the browser with KaTeX.
"""
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site"
SKIP_DIRS = {".git", ".venv", "venv", "site", "node_modules", "__pycache__", ".github"}
SKIP_ASSET_SUFFIXES = {".pdf", ".epub", ".html"}  # built artifacts, not sources


def gh_slug(text):
    # GitHub's heading anchor: lowercase, drop punctuation, spaces to dashes
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def render_math(latex, options):
    tag = "div" if options.get("display_mode") else "span"
    cls = "math-display" if options.get("display_mode") else "math-inline"
    return f'<{tag} class="{cls}">{html.escape(latex)}</{tag}>'


md = (
    MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False})
    .use(anchors_plugin, min_level=1, max_level=6, permalink=False, slug_func=gh_slug)
    # allow_digits=False keeps prices like $5 or $1.39 as plain text, matching the PDF build
    .use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True, renderer=render_math)
    .enable(["table", "strikethrough"])
)


def walk(suffix_filter):
    # only files git knows about (tracked or new, not ignored), so build leftovers never get published
    try:
        names = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):  # no git (e.g. a CI checkout without .git): take everything
        names = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    for name in sorted(names):
        p = ROOT / name
        if p.is_file() and not (set(Path(name).parts) & SKIP_DIRS) and suffix_filter(p):
            yield p


def page_order(md_files):
    # reading order from chapters-md.txt (the book's build order), then everything else
    listed = [ROOT / l.strip() for l in (ROOT / "chapters-md.txt").read_text().splitlines() if l.strip()]
    listed = [p for p in listed if p.exists()]
    rest = [p for p in md_files if p not in set(listed)]
    return listed, rest


# pages that came from Stas Bekman's original repo (the markdown files present when it was forked);
# the sidebar marks them "(Stas)" to tell them apart from the chapters written for this fork
STAS_PAGES = {ROOT / l.strip() for l in (Path(__file__).parent / "stas_pages.txt").read_text().splitlines() if l.strip()}


def nav_label(p, title):
    return f"{title} (Stas)" if p in STAS_PAGES else title


def sidebar_groups(listed, rest, titles):
    """Sidebar sections = the parts of the README's table of contents.

    Each chapter linked under "**Part N. ...**" goes in that part; pages in chapters-md.txt that the
    table of contents doesn't link (sub-pages of a section) join the part of the chapter before them.
    Pages outside the reading order are grouped by top-level folder.
    """
    readme = (ROOT / "README.md").read_text()
    toc = readme[readme.find("## Table of Contents"):readme.find("### Reading paths")]
    part_of, part_names, part = {}, [], None
    for line in toc.splitlines():
        m = re.match(r"\*\*(Part [^*]+)\*\*", line.strip())
        if m:
            part = re.sub(r"\s*-\s*$", "", m.group(1)).rstrip(".")
            part_names.append(part)
            continue
        for link in re.findall(r"\]\(([^)#]+)", line):
            target = (ROOT / link).resolve()
            if target.is_dir():
                target = target / "README.md"
            if part and target.suffix == ".md":
                part_of.setdefault(target, part)
    groups = {"Start here": [], **{n: [] for n in part_names}}
    current = "Start here"
    for p in listed:
        current = part_of.get(p.resolve(), current)
        groups[current].append((p, nav_label(p, titles[p])))
    others = {}
    for p in rest:
        top = p.relative_to(ROOT).parts[0] if len(p.relative_to(ROOT).parts) > 1 else "top level"
        others.setdefault(top, []).append((p, f"{nav_label(p, titles[p])} - {p.relative_to(ROOT)}"))
    out = [(n, pages) for n, pages in groups.items() if pages]
    out += [(f"Other pages: {k}", v) for k, v in sorted(others.items())]
    return out


def title_of(path):
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("# "):
            return re.sub(r"[`*_]", "", line[2:]).strip()
    return str(path.relative_to(ROOT))


def out_path(md_path):
    return OUT / md_path.relative_to(ROOT).with_suffix(".html")


def rewrite_links(body, src):
    """Point links at .md files / directories to the generated .html pages."""
    def fix(m):
        attr, url = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#|data:|/)", url):
            return m.group(0)
        path, _, anchor = url.partition("#")
        target = (src.parent / path).resolve()
        if target.is_dir() and (target / "README.md").exists():
            target = target / "README.md"
        if target.suffix == ".md":
            rel = Path(*[".."] * (len(src.relative_to(ROOT).parts) - 1), target.relative_to(ROOT).with_suffix(".html"))
            url = str(rel) + (f"#{anchor}" if anchor else "")
        return f'{attr}="{url}"'
    return re.sub(r'(href|src)="([^"]+)"', fix, body)


TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - Machine Learning Engineering</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/styles/github-dark.min.css">
<link rel="stylesheet" href="{root}assets/style.css">
</head><body>
<button id="menu" aria-label="toggle navigation">&#9776;</button>
<nav id="sidebar">
  <a class="home" href="{root}index.html">Machine Learning Engineering</a>
  <input id="filter" type="search" placeholder="Filter pages...">
  {nav}
</nav>
<main>
<article>
{body}
</article>
<footer>{prevnext}</footer>
</main>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/highlight.min.js"></script>
<script src="{root}assets/site.js"></script>
</body></html>
"""


def nav_html(groups, current, root):
    parts = []
    for name, pages in groups:
        items = "".join(
            '<li><a href="{}{}"{}>{}</a></li>'.format(root, p.relative_to(ROOT).with_suffix(".html"),
                                                    ' class="current"' if p == current else "", html.escape(t))
            for p, t in pages)
        open_attr = " open" if any(p == current for p, _ in pages) else ""
        parts.append(f"<details{open_attr}><summary>{html.escape(name)}</summary><ul>{items}</ul></details>")
    return "\n".join(parts)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    md_files = list(walk(lambda p: p.suffix == ".md"))
    listed, rest = page_order(md_files)
    titles = {p: title_of(p) for p in md_files}
    sequence = listed + rest
    groups = sidebar_groups(listed, rest, titles)

    # static assets referenced by the markdown (images, scripts, configs)
    for p in walk(lambda p: p.suffix != ".md" and p.suffix not in SKIP_ASSET_SUFFIXES and not p.name.startswith(".")):
        dst = OUT / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).parent / "style.css", assets / "style.css")
    shutil.copy2(Path(__file__).parent / "site.js", assets / "site.js")

    for i, src in enumerate(sequence):
        depth = len(src.relative_to(ROOT).parts) - 1
        root = "../" * depth
        text = src.read_text(errors="ignore")
        front = ""
        m = re.match(r"---\n(.*?)\n---\n", text, flags=re.S)
        if m:  # YAML front matter: GitHub shows it as metadata, not as a heading
            front = f'<pre class="front-matter"><code>{html.escape(m.group(1))}</code></pre>\n'
            text = text[m.end():]
        body = front + rewrite_links(md.render(text), src)
        prev_link = next_link = ""
        if i > 0:
            p = sequence[i - 1]
            prev_link = f'<a class="prev" href="{root}{p.relative_to(ROOT).with_suffix(".html")}">&larr; {html.escape(titles[p])}</a>'
        if i + 1 < len(sequence):
            n = sequence[i + 1]
            next_link = f'<a class="next" href="{root}{n.relative_to(ROOT).with_suffix(".html")}">{html.escape(titles[n])} &rarr;</a>'
        page = TEMPLATE.format(title=html.escape(titles[src]), root=root, nav=nav_html(groups, src, root),
                               body=body, prevnext=prev_link + next_link)
        dst = out_path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(page)

    shutil.copy2(out_path(ROOT / "README.md"), OUT / "index.html")
    (OUT / "pages.json").write_text(json.dumps({str(p.relative_to(ROOT)): titles[p] for p in sequence}, indent=1))
    print(f"wrote {len(sequence)} pages ({len(listed)} in reading order, {len(rest)} others) to {OUT}")


if __name__ == "__main__":
    main()
