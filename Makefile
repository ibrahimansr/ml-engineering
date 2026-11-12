# usage: make help

.PHONY: help check-style spell prep-html-files html html-local pdf epub fix-tables check-links-local check-links-all check-links-local-fast check-redirects check-programs clean
.DEFAULT_GOAL := help

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# pip install codespell
spell: ## spellcheck
	@codespell --write-changes --skip "*.pdf" --skip "*.json"

prep-html-files: ## prepare html-files
	echo book-front.html > chapters-html.txt
	perl -ne 's|\.md|.html|; print' chapters-md.txt >> chapters-html.txt

html: prep-html-files ## make html version w/ scripts linking to their url at my github repo
	python build/mdbook/md-to-html.py

html-local: prep-html-files ## make html version w/ scripts remaining local
	python build/mdbook/md-to-html.py --local

pdf: html ## make pdf version (from html files)
	prince --no-author-style -s build/prince_style.css --pdf-title="Machine Learning Engineering ($$(date))" -o out1.pdf $$(cat chapters-html.txt | tr "\n" " ")
	pdftk out1.pdf dump_data output pdf-bookmarks.txt
	pdftk out1.pdf cat 2-end 1 output out2.pdf
	pdftk out2.pdf update_info pdf-bookmarks.txt output "Machine Learning Engineering.pdf"

epub: html ## make epub version (from html files)
	python build/mdbook/preprocess-html-for-epub.py && \
	pandoc --from html --to epub3 \
		--output "Machine Learning Engineering.epub" \
		--metadata title="Machine Learning Engineering" \
		--metadata date="$$(date +%Y-%m-%d)" \
		--metadata language="en" \
		--resource-path=.:$$(cat chapters-html.txt | xargs -n1 dirname | awk '!seen[$$0]++' | tr "\n" ":") \
		$$(cat chapters-html.txt | tr "\n" " ")

fix-tables: ## fix markdown tables so they render (join multi-line headers, add missing blank lines, realign pipes)
	python build/fix-tables.py

check-links-local: html-local ## check local links
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") | tee linkchecker-local.txt

check-links-all: html ## check all links including external ones
	linkchecker --config build/linkcheckerrc $$(cat chapters-html.txt | tr "\n" " ") --check-extern --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0" | tee linkchecker-all.txt

check-programs: ## check the book's main programs still start up and print --help (doesn't run them)
	@build/check-programs

check-style: ## report hard-wrapped prose, and tables wider than the PDF page
	@python build/check-style.py
	@python build/check-table-width.py

check-links-local-fast: ## scan local links+anchors without building html (no markdown_it needed)
	@python build/check-links.py

check-redirects: ## report external links that have moved, so the book can cite the endpoint (needs network, slow)
	@python build/check-redirects.py

clean: ## remove build files
	find . -name "*html" -exec rm {} \;
