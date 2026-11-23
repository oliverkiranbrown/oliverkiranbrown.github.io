// Footnotes — progressive enhancement.
//
// Markdown first renders as a classic list of footnotes at the bottom of the
// page. Here we move each note inline, directly after the reference that
// points at it, and let you tap the reference to show or hide the note.
// Once this script runs the bottom list is hidden via `html.js-footnotes`
// (see the footnote section in css/index.css).
(function () {
	const refs = document.querySelectorAll("a.footnote-ref[href^='#fn:']");
	if (!refs.length) return;

	document.documentElement.classList.add("js-footnotes");

	function setOpen(note, ref, open) {
		note.classList.toggle("open", open);
		ref.setAttribute("aria-expanded", String(open));
	}

	for (const ref of refs) {
		const source = document.getElementById(ref.getAttribute("href").slice(1));
		if (!source) continue;

		const note = document.createElement("span");
		note.className = "sidenote";
		note.id = "sidenote:" + ref.id.replace(/^fnref:/, "");
		for (const node of source.childNodes) {
			note.append(node.cloneNode(true));
		}
		// A note that is referenced more than once gets one back-link per
		// reference; keep only the one that points back at this reference.
		for (const backref of note.querySelectorAll("a.footnote-backref")) {
			if (backref.getAttribute("href") !== "#" + ref.id) backref.remove();
		}
		ref.after(note);

		ref.addEventListener("click", (event) => {
			event.preventDefault();
			setOpen(note, ref, !note.classList.contains("open"));
		});

		const backref = note.querySelector("a.footnote-backref");
		if (backref) {
			backref.addEventListener("click", (event) => {
				event.preventDefault();
				setOpen(note, ref, false);
				ref.focus();
			});
		}
	}
})();
