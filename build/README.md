# Book Building

Important: this is still a WIP - it mostly works, but stylesheets need some work to make the pdf really nice. Should be complete in a few weeks.

This document assumes you're working from the root of the repo.

## Installation requirements

1. Install python packages used during book build
```bash
pip install -r build/requirements.txt
```

2. Download the free version of [Prince XML](https://www.princexml.com/download/). It's used to build the pdf version of this book.


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

It will first build the html target and then will use it to build the epub version.


## Check links and anchors

To validate that all local links and anchored links are valid run:
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

## Resize images

When included images are too large, make them smaller a bit:

```bash
mogrify -format png -resize 1024x1024\> *png
```


## Build the website

Every markdown file in the repo becomes an HTML page, with a sidebar in reading order, equations typeset by KaTeX and code highlighting:

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

To deploy on [Vercel](https://vercel.com): import the GitHub repo in the Vercel dashboard. `vercel.json` tells Vercel to serve `site/` as-is, with no build step. Or deploy straight from your machine with `npx vercel deploy site --prod`.

## Equations

Write math with `$...$` (inline) and `$$...$$` (display, on their own lines), the same syntax GitHub renders. The pdf build converts it to MathML and the website renders it with KaTeX. Escape literal dollar signs in prose as `\$`. Check a file with:

```bash
python build/check-math.py path/to/chapter.md
```
