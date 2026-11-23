#!/usr/bin/env node
// Prepares a raw Obsidian note as a post for this site.
//
//   node prep-note.mjs <note.md> [--vault <dir>] [--dest <dir>] [--slug <slug>]
//
// What it does:
//   - finds every ![[image]] the note references inside the vault, copies it
//     into the post folder under a URL-safe name, and rewrites the link to match
//   - removes %%comments%% and ^block-ids, which would otherwise render as
//     literal text
//   - joins wrapped footnote definition lines, which the site's `breaks: true`
//     setting would otherwise turn into hard <br>s mid-sentence
//   - writes the front matter, and reports the Obsidian syntax that needs a
//     human decision ([[wiki links]], ==highlights==, task lists, #inline-tags)
//
// Two modes:
//   no --dest   dry run — report goes to stderr, prepped markdown to stdout
//   --dest dir  do it — images are copied into dir, and <slug>.md is written there
//
// The vault path can live in config.json beside this script's parent directory,
// so that a bare `--dest content/writing/foo` is enough. Flags win over config.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function readConfig(configPath) {
	try {
		return JSON.parse(fs.readFileSync(configPath, "utf8"));
	} catch {
		return {};
	}
}

const here = path.dirname(fileURLToPath(import.meta.url));
const config = readConfig(path.join(here, "..", "config.json"));

const args = process.argv.slice(2);
const value = (name) => {
	const i = args.indexOf(name);
	return i === -1 ? undefined : args[i + 1];
};
const positional = [];
for (let i = 0; i < args.length; i++) {
	if (args[i].startsWith("--")) {
		if (["--vault", "--dest", "--slug"].includes(args[i])) i++;
		continue;
	}
	positional.push(args[i]);
}

const file = positional[0];
if (!file) {
	console.error("usage: node prep-note.mjs <note.md> [--vault <dir>] [--dest <dir>] [--slug <slug>]");
	process.exit(1);
}

const vault = value("--vault") || config.vault;
const destDir = value("--dest") ? path.resolve(value("--dest")) : null;
const noteDir = path.dirname(path.resolve(file));
const slug = (value("--slug") || path.basename(file, ".md"))
	.toLowerCase()
	.replace(/[^a-z0-9]+/g, "-")
	.replace(/^-+|-+$/g, "");

let src = fs.readFileSync(file, "utf8");
const applied = [];
const todo = [];
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// --- Vault index --------------------------------------------------------------
// Obsidian lets a note reference an attachment by name from anywhere in the
// vault, so one by-basename index of the vault is what resolves ![[image.png]].
const vaultIndex = new Map();
function indexVault(dir) {
	let entries;
	try {
		entries = fs.readdirSync(dir, { withFileTypes: true });
	} catch {
		return;
	}
	for (const entry of entries) {
		if (entry.name.startsWith(".")) continue;
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) indexVault(full);
		else if (entry.isFile() && !vaultIndex.has(entry.name.toLowerCase())) vaultIndex.set(entry.name.toLowerCase(), full);
	}
}
if (vault) {
	if (fs.existsSync(vault)) indexVault(vault);
	else todo.push(`the vault path "${vault}" does not exist — pass --vault <dir>`);
}

// --- Finding and placing images ----------------------------------------------
const IMAGE = /\.(png|jpe?g|gif|webp|avif|svg|bmp|tiff?)$/i;
const safeName = (name) => name.replace(/\s+/g, "-").replace(/[#?%:]+/g, "");

// Absolute paths outside the working directory are more readable than ../../../
const show = (target) => {
	const relative = path.relative(process.cwd(), target);
	return relative.startsWith("..") ? target : relative;
};

function findAsset(reference) {
	const clean = decodeURIComponent(reference.trim());
	const near = [path.resolve(noteDir, clean), path.resolve(noteDir, path.basename(clean))];
	for (const candidate of near) {
		if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
	}
	return vaultIndex.get(path.basename(clean).toLowerCase()) || null;
}

// Copies an asset into the post folder and returns the filename to link to.
function place(assetPath) {
	const name = safeName(path.basename(assetPath));
	if (!destDir) return name;
	fs.mkdirSync(destDir, { recursive: true });
	const target = path.join(destDir, name);
	if (fs.existsSync(target)) {
		if (!fs.readFileSync(target).equals(fs.readFileSync(assetPath))) {
			todo.push(`"${name}" is already in the post folder with different content — decide which one to keep`);
		}
		return name;
	}
	fs.copyFileSync(assetPath, target);
	applied.push(`copied ${path.basename(assetPath)} → ${show(target)}`);
	return name;
}


// --- Obsidian comments -------------------------------------------------------
// A comment on its own line goes with the whole line. One embedded mid-paragraph
// leaves a single space, so the text either side of it rejoins instead of
// leaving a line break behind.
const commentCount = (src.match(/%%[\s\S]*?%%/g) || []).length;
if (commentCount) {
	src = src.replace(/^[ \t]*%%[\s\S]*?%%[ \t]*\r?\n/gm, "");
	src = src.replace(/%%[\s\S]*?%%/g, "\u0000").replace(/\s*\u0000\s*/g, " ");
	applied.push(`deleted ${plural(commentCount, "%%comment%% block")}`);
}

// --- Block references --------------------------------------------------------
const blockRefs = src.match(/[ \t]+\^[A-Za-z0-9-]+[ \t]*$/gm) || [];
if (blockRefs.length) {
	src = src.replace(/[ \t]+\^[A-Za-z0-9-]+[ \t]*$/gm, "");
	applied.push(`deleted ${plural(blockRefs.length, "^block-id")}`);
}

// --- Embeds: ![[image.png]] / ![[image.png|alt]] -----------------------------
let embeds = 0;
src = src.replace(/!\[\[([^\]|]+?)(?:\|([^\]]*))?\]\]/g, (_match, target, alias) => {
	const clean = target.trim();
	const name = path.basename(clean);
	const alt = (alias || name.replace(/\.[a-z0-9]+$/i, "")).trim();

	if (!IMAGE.test(name)) {
		todo.push(`![[${clean}]] is not an image — embeds are not supported, inline it or link to it`);
		return `![[${clean}]]`;
	}

	const found = findAsset(clean);
	embeds++;
	if (!found) {
		todo.push(`could not find "${name}"${vault ? " in the vault" : " — pass --vault <dir> so it can be looked up"}`);
		return `![${alt}](./${safeName(name)})`;
	}
	return `![${alt}](./${place(found)})`;
});
if (embeds) applied.push(`rewrote ${plural(embeds, "![[embed]]")} as markdown images`);

// --- Markdown images pointing somewhere else ---------------------------------
let relinked = 0;
src = src.replace(/(!\[[^\]]*\]\()(?!https?:|mailto:|\/|\.\/|#)([^)\s]+)\)/gi, (_match, pre, target) => {
	const name = path.basename(decodeURIComponent(target));
	if (!IMAGE.test(name)) return `${pre}${target})`;

	const found = findAsset(target);
	relinked++;
	if (!found) {
		todo.push(`could not find "${name}" — copy it into the post folder yourself`);
		return `${pre}./${safeName(name)})`;
	}
	return `${pre}./${place(found)})`;
});
if (relinked) applied.push(`pointed ${plural(relinked, "image link")} at a file beside the post`);

// --- Footnotes: join wrapped lines -------------------------------------------
const FOOTNOTE_DEF = /^\[\^[^\]]+\]:/;
const BLANK = /^[ \t]*$/;
const BLOCK_START = /^[ \t]*([-*+]\s|\d+[.)]\s|>|\||```|~~~|#{1,6}\s|<)/;

let joined = 0;
const out = [];
const lines = src.split("\n");
for (let i = 0; i < lines.length; i++) {
	if (!FOOTNOTE_DEF.test(lines[i])) {
		out.push(lines[i]);
		continue;
	}

	// A definition absorbs the lazy continuation lines that follow it, so fold
	// them onto one line before `breaks: true` can insert a <br>.
	let text = lines[i].trimEnd();
	while (
		i + 1 < lines.length &&
		!BLANK.test(lines[i + 1]) &&
		!FOOTNOTE_DEF.test(lines[i + 1]) &&
		!BLOCK_START.test(lines[i + 1])
	) {
		text += " " + lines[i + 1].trim();
		joined++;
		i++;
	}
	out.push(text);
}
if (joined) {
	src = out.join("\n");
	applied.push(`joined ${plural(joined, "wrapped footnote line")} so they do not become hard line breaks`);
}

// --- Syntax this site cannot render ------------------------------------------
const count = (re) => (src.match(re) || []).length;

const wikiLinks = count(/(?<!!)\[\[[^\]]+\]\]/g);
if (wikiLinks) todo.push(`${plural(wikiLinks, "[[wiki link]]")} left as-is — there is no wiki-link support, rewrite as [text](/writing/slug/)`);

const highlights = count(/==[^=\n]+==/g);
if (highlights) todo.push(`${plural(highlights, "==highlight==")} left as-is — use **bold** or *italic*`);

const tasks = count(/^[ \t]*[-*+] \[[ xX]\]/gm);
if (tasks) todo.push(`${plural(tasks, "task list item")} left as-is — they render as literal "[ ]" text`);

const inlineTags = count(/(?:^|\s)#[A-Za-z][\w/-]*/gm);
if (inlineTags) todo.push(`${plural(inlineTags, "#inline-tag")} left as-is — move them into the front matter tags`);

// --- Front matter -------------------------------------------------------------
// An Obsidian note usually opens with its own H1 title. The post layout already
// renders the title from the front matter, so that heading becomes the title and
// comes out of the body.
let frontMatter = "";
if (!/^---\r?\n/.test(src)) {
	const bodyLines = src.split("\n");
	const firstContent = bodyLines.findIndex((line) => line.trim() !== "");
	let title = slug.replace(/-/g, " ");
	if (firstContent !== -1 && /^#\s+\S/.test(bodyLines[firstContent])) {
		title = bodyLines[firstContent].replace(/^#\s+/, "").trim();
		bodyLines.splice(firstContent, /^\s*$/.test(bodyLines[firstContent + 1] ?? "") ? 2 : 1);
		src = bodyLines.join("\n");
		applied.push("used the opening H1 as the title, and removed it from the body");
	}

	frontMatter = [
		"---",
		`title: ${title}`,
		"description:",
		"tags: ['Writing']",
		`date: ${new Date().toISOString().slice(0, 10)}`,
		"layout: layouts/post.njk",
		"eleventyNavigation:",
		`  key: ${slug}`,
		"  parent: writing",
		"---",
		"",
		"",
	].join("\n");
}

// --- Report -------------------------------------------------------------------
console.error(`Note:  ${show(path.resolve(file))}`);
console.error(`Slug:  ${slug}`);
if (vault) console.error(`Vault: ${vault}`);
if (destDir) console.error(`Dest:  ${show(destDir)}`);
console.error("");
console.error(applied.length ? `Applied:\n  - ${applied.join("\n  - ")}` : "Applied:\n  - nothing needed changing");
console.error("");
console.error(todo.length ? `Still to do:\n  - ${todo.join("\n  - ")}` : "Still to do:\n  - nothing flagged");
console.error("");

// --- Write --------------------------------------------------------------------
if (destDir) {
	fs.mkdirSync(destDir, { recursive: true });
	const outFile = path.join(destDir, `${slug}.md`);
	fs.writeFileSync(outFile, frontMatter + src);
	console.error(`Wrote ${show(outFile)}`);
	if (frontMatter) console.error("Front matter was added — set the description and the tags before committing.");
} else {
	process.stdout.write(frontMatter + src);
}

