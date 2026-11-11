// equations: the build leaves raw LaTeX in .math-inline / .math-display; KaTeX typesets it here
document.querySelectorAll(".math-inline, .math-display").forEach(function (el) {
  try {
    katex.render(el.textContent, el, { displayMode: el.classList.contains("math-display"), throwOnError: false });
  } catch (e) { /* leave the raw LaTeX visible */ }
});

// code highlighting for fenced blocks that declare a language
document.querySelectorAll('pre code[class*="language-"]').forEach(function (el) { hljs.highlightElement(el); });

// sidebar: filter pages by title, toggle on small screens, keep the current page in view
var filter = document.getElementById("filter");
filter.addEventListener("input", function () {
  var q = filter.value.toLowerCase();
  document.querySelectorAll("#sidebar li").forEach(function (li) {
    li.style.display = li.textContent.toLowerCase().includes(q) ? "" : "none";
  });
  if (q) document.querySelectorAll("#sidebar details").forEach(function (d) { d.open = true; });
});
document.getElementById("menu").addEventListener("click", function () { document.body.classList.toggle("nav-open"); });
// scroll only the sidebar (scrollIntoView would also scroll the page sideways when content is wide)
var sidebar = document.getElementById("sidebar");
var current = document.querySelector("#sidebar a.current");
if (current) sidebar.scrollTop = current.offsetTop - sidebar.clientHeight / 2;
