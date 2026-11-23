---
name: obsidian-post
description: Use when turning a raw note written in Obsidian into a post for this Eleventy site — invoked as "/post <path to a note in my vault>". Finds every image the note references anywhere in the vault, copies them into the post folder and rewrites the links, writes the YAML front matter, and formats footnotes so they expand inline instead of breaking. Also use when asked to "add a new post", "publish this note", "prep this markdown for the site", or to fix a note that is already under content/writing.
version: 1.0.0
user-invocable: true
argument-hint: "<path to a note in your vault>"
license: MIT
allowed-tools:
  - Bash(node .claude/skills/obsidian-post/scripts/*)
  - Bash(npx @11ty/eleventy *)
  - Bash(ls *)
  - Bash(mkdir *)
  - Bash(cp *)
  - Bash(mv *)
  - Bash(./dewhite.sh *)
---

Turns the raw markdown you paste out of Obsidian into a post this site renders properly. The content is never rewritten for style — only the front matter is added, the attachments are relocated and relinked, and the handful of syntaxes Obsidian invented that the renderer does not know about are converted.

## Target layout

```
content/writing/<slug>/<slug>.md      the post
content/writing/<slug>/<image>.png    its images, beside it
```

The directory name and the file name must match, because that is what gives the post its URL: `content/writing/foo/foo.md` is served at `/writing/foo/`. Use lowercase with hyphens or underscores, no spaces.

## The workflow

When asked to post a note, either as `/post <path>` or in prose:

1. **Get the vault path.** Look for `config.json` beside this file. If it is not there, ask for the vault root once, then write it in (copy `config.example.json` first) so later runs need no prompting. `config.json` is gitignored because the path is machine-specific.
2. **Run the prep script** — it performs every step marked ⚙️ below:

   ```bash
   node .claude/skills/obsidian-post/scripts/prep-note.mjs \
     "<vault>/<note>.md" \
     --vault "<vault root>" \
     --dest content/writing/<slug>
   ```

   Without `--dest` it is a dry run: the report goes to stderr, the prepped markdown to stdout. With `--dest` it copies the images into that folder and writes `<slug>.md` there.
3. **Read its report, and act on `Still to do:`** — that is the list of things it deliberately refuses to guess: wiki links to rewrite, highlights to convert, task lists, inline tags, and any image it could not find in the vault. None of them should be left outstanding.
4. **Fill in the front matter** it generated. At minimum `description`, and the `tags` beyond the `'Writing'` default.
5. **Build and actually look at the page** — step 5 below.
6. **Commit**, and ask before pushing — see step 6.

Everything below is the detail behind those steps. Steps 1 to 4 describe what the script does and what to do by hand when it cannot.

## 1. Front matter

The script writes this for you, taking `title` from the note's opening `# H1` (and removing that heading, because the layout renders the title itself). What it cannot know is the `description` and the right `tags`, so fill those in. The shape it produces:

```yaml
---
title: Sentence case title
description: One line of context, used for the page meta description and list previews.
tags: ['Writing', 'Topic']
date: YYYY-MM-DD
layout: layouts/post.njk
eleventyNavigation:
  key: <slug>
  parent: writing
---
```

- `tags` — always include `'Writing'`; add a topic. In use today: `Writing`, `Personal`, `ML`, `Coding`, `Experimental`.
- `date` — `YYYY-MM-DD`. Drives ordering in the writing list.
- `layout` and the `posts` tag are already applied to everything under `content/writing/` by `content/writing/blog.11tydata.js`, so they are optional; every existing post sets `layout` explicitly anyway.
- `draft: true` renders the post with `npm start` but keeps it out of a real build.
- `eleventyNavigation` puts the post in the top bar; `key` should be the slug and `parent: writing`.
- For a plain page with no post chrome (no date, tags or previous/next links), use `layout: layouts/about.njk`.

## 2. Images ⚙️

Obsidian stores attachments wherever you configured it, and a note can reference one by bare filename from anywhere in the vault. The images must end up **beside the post file**, because the build only optimises images it can resolve relative to the page.

The script handles this when given `--vault`: it indexes the vault by filename, then for every `![[image]]` it copies the file into `content/writing/<slug>/`, renames it, and rewrites the link. Use the manual steps below only for what it reports under `Still to do:`, or when working by hand.

1. Copy the image into `content/writing/<slug>/`.
2. Rename it to something URL-safe: no spaces, and no `#`, `?`, `:` or `%`. `Screenshot 2026-01-02 at 14.03.11.png` becomes `screenshot-2026-01-02.png`. Spaces have to be written as `%20` in the link to work (`BT Site Diagram-1.jpg` is referenced as `BT%20Site%20Diagram-1.jpg`), which is easy to get wrong — renaming is the better fix.
3. Rewrite the link and keep meaningful alt text:

```md
![[Screenshot 2026-01-02 at 14.03.11.png]]     →   ![Screenshot of the dashboard](./screenshot-2026-01-02.png)
![](attachments/diagram.png)                   →   ![The request flow](./diagram.png)
```

Then leave the image alone: `@11ty/eleventy-img` converts each one to `avif`, `webp` and the original format, adds `srcset`, `width` and `height`, and lazy-loads it. Do not pre-convert, hand-write `<picture>` markup, or move images into `public/` — that folder is copied verbatim and never optimised (it holds the favicon, fonts and drum samples).

Remote images do work — the build fetches and optimises them — but that needs network access at build time and slows builds down, so download them first when you can.

Optional: if an image has a white background and the page background is not white, `./dewhite.sh content/writing/<slug>` recolours near-white pixels to the page colour. Set `TARGET_COLOR` in the script to the current `--background-color` first. The background is currently white, so this is usually unnecessary.

## 3. Footnotes ⚙️

Keep writing them exactly as Obsidian does — no conversion needed:

```md
A claim that needs a source.[^1] And a second one.[^2]

## Footnotes

[^1]: The note.
[^2]: Another note, with a second paragraph:

    Indented four spaces.
```

The `## Footnotes` heading is stripped automatically, so it can stay where it is. Two rules matter:

- **One line per note.** This site sets `breaks: true`, so a source line that wraps becomes a hard `<br>` in the middle of the rendered sentence. The prep script joins wrapped note lines for you.
- **A second paragraph** needs a blank line and a four-space indent, not just an indent.

Worth knowing when the numbers look wrong: the displayed number follows the order the references appear in the text, not the label written, so a `[^4]` referenced before `[^3]` shows as `3`. A note referenced twice keeps one number and gets a back-link per reference. Inline notes (`^[like this]`) are also supported.

## 4. Obsidian syntax the renderer does not know ⚙️

Each of these was rendered through this site's actual markdown pipeline to check:

| Written in Obsidian | What this site does with it | What to do instead |
| --- | --- | --- |
| `[[Wiki link]]`, `![[Some note]]` | renders literally, brackets and all | rewrite as `[text](/writing/other-slug/)`, or paste in the embedded content |
| `==highlight==` | renders literally | use `**bold**` or `*italic*` |
| `%%private comment%%` | renders literally | delete it |
| `^block-id` | renders literally | delete it |
| `#inline-tag` | renders literally | move it into the front matter `tags` |
| `- [ ] task item` | renders as literal `[ ]` text | use a plain bullet, unless you want the literal brackets |
| `> [!note] Title` callout | **works** — a styled alert box | leave it alone |
| `$inline$` and `$$display$$` maths | **works**, rendered with MathJax | leave it alone |
| tables, `> quotes`, `---`, code fences, headings, lists | work | leave them alone |

The wrapping rule from the footnotes section applies to the whole document: write each paragraph on one line, because a line that wraps becomes a `<br>`. That is why not one post in `content/writing/` contains a `<br>`.

## 5. Verify before handing it over

```bash
npx @11ty/eleventy --output=/tmp/preview   # build to somewhere disposable
npm start                                  # or serve it and click around
```

Then check the built page rather than the source:

- every image resolved, and its `<img>` has a `srcset` — that is the signal the image transform picked it up
- tapping a footnote reference expands the note in place, and nothing jumped to the bottom of the page
- no leftover `[[`, `==`, `%%`, `^id` or `]](` anywhere in the output
- no `<br>` in the middle of a sentence

One caveat: `npm start` writes into `_site/`, and `_site/` is tracked in git in this repository. Tidy up afterwards with `git checkout -- _site`.

## 6. Committing

```bash
git add content/writing/<slug>
git commit -m "Add <title>"
```

Then ask before pushing — and check where you are first:

- `git branch --show-current`. On a branch other than `main`, pushing the branch publishes nothing, so offer a PR instead (`gh pr create`).
- **Pushing to `main` deploys the live site**, because `.github/workflows/deploy-to-ghpages.yml` runs on every push to `main`. Confirm with the user before doing it, and never force-push `main`.
- **Never commit `_site/`.** It is tracked here but it is build output, and a stray `npm start` leaves it dirty — `git checkout -- _site` restores it. Stage only the paths you actually added or changed.

