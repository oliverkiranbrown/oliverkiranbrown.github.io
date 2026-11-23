// Footnotes — Obsidian/Pandoc `[^1]` and `[^1]:` syntax, rendered the way
// hermit-tech.com does it: superscript references, a hanging-indent list at
// the end of the post, and (with JavaScript) notes that expand inline next
// to the reference that points at them.
import markdownItFootnote from "markdown-it-footnote";

// Visible marker for a footnote, e.g. 1. This is also the id of its entry in
// the list at the end of the post, so every reference to the same note points
// at the same anchor.
function footnoteNumber(token) {
	return Number(token.meta.id + 1);
}

// Fragment id for a single reference: "1", or "1:2" for a second reference to
// footnote 1. Keeps each reference, and its back-link, unique.
function referenceId(token) {
	const n = footnoteNumber(token);
	return token.meta.subId > 0 ? `${n}:${token.meta.subId}` : `${n}`;
}

export function footnotes(md) {
	md.use(markdownItFootnote);

	// <a id="fnref:1" href="#fn:1" class="footnote-ref"><sup>1</sup></a>
	md.renderer.rules.footnote_ref = (tokens, idx) => {
		const n = footnoteNumber(tokens[idx]);
		return `<a id="fnref:${referenceId(tokens[idx])}" href="#fn:${n}" class="footnote-ref" role="doc-noteref" aria-expanded="false"><sup>${n}</sup></a>`;
	};

	md.renderer.rules.footnote_block_open = () =>
		`<section class="footnotes" role="doc-endnotes">\n<ol class="footnotes-list">\n`;
	md.renderer.rules.footnote_open = (tokens, idx) =>
		`<li id="fn:${footnoteNumber(tokens[idx])}" class="footnote-item" role="doc-endnote">`;
	md.renderer.rules.footnote_anchor = (tokens, idx) =>
		` <a href="#fnref:${referenceId(tokens[idx])}" class="footnote-backref" role="doc-backlink" aria-label="Return to the text">\u21a9\uFE0E</a>`;
	md.renderer.rules.footnote_close = () => `</li>\n`;
	md.renderer.rules.footnote_block_close = () => `</ol>\n</section>\n`;
}

// Obsidian keeps its footnote definitions under a trailing `# Footnotes`
// heading. markdown-it-footnote moves those definitions to the end of the
// document itself, which would leave the heading stranded. Remove the heading,
// but only when the file really does contain `[^x]:` definitions, so that a
// document which merely mentions footnotes keeps its own heading.
const FOOTNOTE_HEADING = /^[ \t]*#{1,6}[ \t]*footnotes?[ \t]*$/im;
const FOOTNOTE_DEFINITION = /^\[\^[^\]]+\]:/m;

export function stripObsidianFootnotesHeading(content) {
	if (!FOOTNOTE_DEFINITION.test(content)) return content;
	return content.replace(FOOTNOTE_HEADING, "");
}
