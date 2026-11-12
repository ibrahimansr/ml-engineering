# Book Building

Important: this is still a WIP - it mostly works, but stylesheets need some work to make the pdf really nice. Should be complete in a few weeks.

This document assumes you're working from the root of the repo. The `make` targets below are defined in the repo's `Makefile` (see [Make and Makefiles](../foundations/toolkit.md#make-and-makefiles)).

## Installation requirements

1. Install python packages used during book build
```bash
pip install -r build/requirements.txt
```

2. Download the free version of [Prince XML](https://www.princexml.com/download/). It's used to build the pdf version of this book. (Prince converts HTML plus CSS into a PDF.)

3. The `pdf` target also calls `pdftk` (a command line PDF toolkit, used to reorder pages and copy bookmarks), and the `epub` target calls the `pandoc` document converter program. The `pandoc` entry in `build/requirements.txt` is only a Python wrapper, so install the program itself too (e.g. `brew install pandoc pdftk-java` on macOS, `apt install pandoc pdftk` on Ubuntu).


## Build html

```bash
make html
```

## Build pdf

```bash
make pdf
```

It will first build the html target and then will use it to build the pdf version.

## Build epub

```bash
make epub
```

It will first build the html target and then will use it to build the epub version (epub is the e-book format most e-readers use).


## Check links and anchors

To validate that all local links and anchored links (links to a specific heading inside a page, `file.md#heading`) are valid run (both targets use `linkchecker`, installed by `build/requirements.txt`):
```bash
make check-links-local
```

To additionally also check external links
```bash
make check-links-all
```
use the latter sparingly to avoid being banned for hammering servers.


## Move md files/dirs and adjust relative links


e.g. `slurm` => `orchestration/slurm`
```bash
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local

```

Step by step: `git mv` moves the files and records the move in git, `perl -pi -e "s|$src|$dst|"` edits `chapters-md.txt` in place, replacing the first occurrence of the old path on each line with the new one, `mv-links.py` rewrites the relative links in and to the moved files, and `make check-links-local` verifies the result.

## Resize images

When included images are too large, make them smaller a bit:

```bash
mogrify -format png -resize 1024x1024\> *png
```

`mogrify` is ImageMagick's in-place image editor (it overwrites the files). `1024x1024>` fits each image inside a 1024x1024 box keeping the aspect ratio, and the `>` (escaped as `\>` so the shell doesn't treat it as a redirect) means only images larger than that are shrunk.


## Build the website

Every markdown file in the repo becomes an HTML page, with a sidebar in reading order, equations typeset by KaTeX (a JavaScript library that renders LaTeX math in the browser) and code highlighting. `python -m http.server 8000` is Python's built-in static file server, serving the current directory on port 8000:

```bash
pip install -r build/requirements.txt
python build/site/build_site.py          # writes ./site
cd site && python -m http.server 8000    # preview at http://localhost:8000
```

The built `site/` directory is committed, so the repo always carries a ready-to-serve website. After changing any markdown, rebuild and check it before committing:

```bash
python build/site/build_site.py
python build/site/check_site.py   # every page, heading, paragraph, code block, equation, image and link is present
```

To deploy on [Vercel](https://vercel.com) (a static-site and web-app hosting service): import the GitHub repo in the Vercel dashboard. `vercel.json` tells Vercel to serve `site/` as-is, with no build step. Or deploy straight from your machine with `npx vercel deploy site --prod` (`npx` comes with Node.js and runs an npm package's command line tool, here Vercel's CLI, without installing it first).

## Equations

Write math with `$...$` (inline) and `$$...$$` (display, on their own lines), the same syntax GitHub renders. The pdf build converts it to MathML (the XML markup for math that browsers and Prince render natively) and the website renders it with KaTeX. Escape literal dollar signs in prose as `\$`. Check a file with:

```bash
python build/check-math.py path/to/chapter.md
```
