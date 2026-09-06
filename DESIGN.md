---
name: Oliver Kiran Brown — Personal Site
description: A plain, text-first personal notebook for writing, notes, and thinking in progress.
colors:
  sage-olive: "#687a3f"
  neutral-bg: "#ffffff"
  ink: "#333333"
  border-light: "#e0e0e0"
  border-mid: "#c0c0c0"
  link-default: "#082840"
  link-active: "#5f2b48"
  link-visited: "#17050f"
  code-bg: "#f6f8fa"
  code-ink: "#24292e"
  tag-bg: "#f0f0f0"
  tag-bg-hover: "#dddddd"
typography:
  nav:
    fontFamily: "-apple-system, system-ui, sans-serif"
    fontSize: "1.2rem"
  logo:
    fontFamily: "Literata, serif"
    fontWeight: 200
    fontSize: "1.4em"
    note: "1.4em is relative to the header's 1.2rem nav context, so it renders at ~1.68rem (~27px), not 1.4rem. 200 is Literata's minimum weight (the earlier DM Serif Text ran to 100)."
  body:
    fontFamily: "Literata, serif"
    fontWeight: 400
    fontSize: "1rem"
    lineHeight: 1.5
  mono:
    fontFamily: "Consolas, Menlo, Monaco, Andale Mono WT, DejaVu Sans Mono, Courier New, monospace"
    fontSize: "0.95rem"
    lineHeight: 1.5
  label:
    fontFamily: "Literata, serif"
    fontSize: "0.9rem"
    lineHeight: 1.4
rounded:
  sm: "4px"
  md: "6px"
spacing:
  sm: "0.5em"
  md: "1em"
  lg: "2rem"
components:
  header-bar:
    backgroundColor: "{colors.sage-olive}"
    textColor: "#ffffff"
    padding: "1em"
  tag-chip:
    backgroundColor: "{colors.tag-bg}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0.2em 0.5em"
  tag-chip-hover:
    backgroundColor: "{colors.tag-bg-hover}"
  code-block:
    backgroundColor: "{colors.code-bg}"
    textColor: "{colors.code-ink}"
    rounded: "{rounded.md}"
    padding: "1rem"
---

# Design System: Oliver Kiran Brown — Personal Site

## 1. Overview

**Creative North Star: "The Lab Notebook"**

This is a working notebook, not a designed brand page. Every visual decision defers to the writing: a serif built for reading carries the actual content, a narrow column keeps it comfortable, one accent color, and no visual performance of authority — no dashboards of stats, no card grids, no gradients — because the site's entire claim is earnestness: this is someone thinking in public, not a polished portfolio funnel. The green top bar (plain system sans, kept deliberately apart from the reading experience) is the one piece of UI chrome; everything inside the content column reads like a considered, printed page, in the spirit of references like y1d2.com and neelnanda.io.

This system explicitly rejects the generic "data scientist portfolio" template: no hero-metric stat blocks, no identical icon-plus-heading card grids, no gradient accents, no glassmorphism. The plainness is the point, not a placeholder for a future redesign.

**Key Characteristics:**
- A single sage-olive accent carries all brand color; nothing else competes with it.
- Flat by default — no shadows, no cards, no layered surfaces.
- Literata (a serif built for long-form reading) carries the entire content column — headings, body, labels, tags — at weights from 200 (wordmark) to 700 (bold text); nav and footer chrome outside that column stay plain system sans.
- Narrow, centered content column (`max-width: 40em`) so long-form writing stays comfortable to read.
- Header/nav text and icons are always white; hover is signaled by underline or scale, never a dimmer color.
- Page-to-page navigation settles in quietly (native View Transitions, fade + 6px rise) rather than cutting or using the browser's flat default crossfade.

## 2. Colors

A single accent color against a plain white-and-ink neutral base; no secondary or tertiary brand colors exist today.

### Primary
- **Sage Olive** (#687a3f): The site's one brand color. Used for the header bar background, the checked state of the custom tag checkbox, and the reading-progress bar. Deepened from an earlier #8ba254 specifically so white header text clears WCAG AA (4.73:1 vs. 2.84:1) — same hue and saturation, just less light.

### Neutral
- **Paper White** (#ffffff): Page background. Also the header/nav text color — white-on-#687a3f is the only place it's used against a non-white surface.
- **Ink** (#333333): Primary body text color (`--color-gray-90`).
- **Hairline** (#e0e0e0): Dashed dividers (nav-list borders, post next/prev separator).
- **Mid Gray** (#c0c0c0): Reserved secondary neutral (`--color-gray-50`), currently underused.

### Link Colors
- **Deep Navy** (#082840): Default link color — a deliberately quiet, ink-adjacent blue rather than a bright brand blue.
- **Plum Active** (#5f2b48): Hover/active link state.
- **Near-Black Violet** (#17050f): Visited link state.

### Code & Tags
- **Code Surface** (#f6f8fa background / #24292e text): Block code, GitHub-light–derived syntax palette.
- **Tag Chip** (#f0f0f0, hover #dddddd): Flat gray pill for post tags — the one place a "chip" component exists.

### Named Rules
**The One Accent Rule.** Sage Olive is the only saturated color on the page. It appears on the header bar, one interactive checked-state, and the reading-progress bar; it never appears as decoration, gradient, or a second competing brand hue.

**The White-on-Accent Rule.** Header/nav text and icons are always white, never dimmed on hover/active (any shade darker than pure white drops below the 4.5:1 this background needs) — hover is signaled by underline (links) or a small scale-up (social icons), not a color shift.

## 3. Typography

**Display/Logo/Body Font:** Literata (variable, opsz 7–72 / wght 200–900; with serif fallback)
**Nav Font:** -apple-system, system-ui (with sans-serif fallback) — header/footer chrome only
**Mono Font:** Consolas, Menlo, Monaco, DejaVu Sans Mono (with monospace fallback)

**Character:** Serif for reading, sans for navigating. Literata — a variable book serif built for long-form reading (Google/TypeTogether) — carries everything inside the content column: the wordmark, headings, body copy, labels and tags, each at a different point on its 200–900 weight range. Nav links and footer chrome, which sit outside that column, stay in plain system sans so the reading experience and the site's UI furniture read as two distinct registers. Literata replaced an earlier DM Serif Text (wordmark-only) specifically because DM Serif Text is a saturated training-data default; Literata is warmer, built for exactly this job, and far less common as a display face.

### Hierarchy
- **Nav** (1.2rem, system sans): The header bar's own font-size context — nav links and social icons sit at this size, outside the Literata content column.
- **Logo** (200 weight, 1.4em, Literata): The home-link wordmark. Because 1.4em is relative to the header's 1.2rem nav context, it renders at ~1.68rem (~27px), not 1.4rem — Literata's thinnest available weight.
- **Body** (400 weight, 1em, Literata, line-height 1.5): All prose; justified alignment, max line length ~40em (well within the 65–75ch guideline).
- **Post title (h1)** and **section headings (h2–h4)**: Literata, bold, left-aligned, no scale drama — headings are structural, not decorative, but now share the reading column's serif rather than sitting in sans.
- **Label** (0.9rem, line-height 1.4, Literata): Small, quiet, gray-ink text for metadata, tags, and secondary widget content — inherits the content column's serif rather than a separate sans. Post dates use a tighter 0.8125em variant of the same role.
- **Mono** (0.95rem, monospace stack): Inline and block code; explicitly overrides back to monospace since it also sits inside the content column.

### Named Rules
**The Invisible Body Rule.** Body text uses the system font stack, not a licensed webfont. The only place a "chosen" typeface appears is the wordmark; everywhere else, type should get out of the way of the words.

## 4. Elevation

Flat by design, and intentionally so. There are no shadows anywhere in the system — no card elevation, no button lift, no modal layering. Depth, where it exists at all, is conveyed through a single dashed hairline border (`#e0e0e0`) separating sections (nav list, next/prev post links), never through shadow or blur. This is a settled rule, not a placeholder: the earnest, understated personality calls for a page that reads as flat text, not a UI with simulated depth.

### Named Rules
**The Flat-By-Default Rule.** No `box-shadow` anywhere in the system. Separation between elements is handled with a 1px dashed hairline (`border-top: 1px dashed #e0e0e0`) or whitespace, never a shadow.

## 5. Components

Component vocabulary is deliberately small: a header bar, a text link, a tag chip, a code block, and a post-list row. There is no button component in active use, no card, no modal.

### Header Bar
- **Style:** Full-width Sage Olive (#687a3f) bar, white text, flex row with wrapping, `padding: 1em`.
- **Wordmark:** Literata, 200 weight, 1.4em (relative to the header's 1.2rem nav context, so it renders at ~1.68rem), no underline unless hovered.
- **Nav:** Inline flex list, no active-state background, no hover color change — current page is marked with an underline only (`aria-current="page"`), not a pill or highlight.
- **Social icons:** Inline SVGs, white fill (unchanged on hover), scale to 1.12× on hover — no background, no border.

### Links
- **Default:** Deep Navy (#082840), no underline until hover.
- **Hover/Active:** Plum (#5f2b48).
- **Visited:** Near-black violet (#17050f).
- **Behavior:** Underline appears only on hover/active; visited state is tracked but subtle, not a strong visual signal.

### Tag Chip
- **Shape:** 4px radius, flat gray fill (#f0f0f0).
- **Hover:** Darker gray fill (#dddddd), no border or shadow change.
- **Use:** Post metadata tags only; the one place a pill/chip pattern exists in the system.

### Code Block
- **Shape:** 6px radius, `padding: 1rem`, `overflow-x: auto`.
- **Colors:** GitHub-light–derived (#f6f8fa background, #24292e text), Prism token colors layered on top for syntax highlighting.
- **Inline code:** Same ink color, subtle background tint, 6px radius, no border.

### Post List Row
- **Style:** Flex row, wrapping; a bold post title, a small quiet date, optional tag chips — no card wrapper, no border, no background. Rows are separated by margin only.

### Custom Checkbox (tag filter)
- **Shape:** 1em square, 4px radius, 2px border (#ccc default).
- **Checked state:** Fills with Sage Olive (#687a3f) and border to match.

### Reading Progress
- **Style:** 3px fixed bar at the top of the viewport, Sage Olive fill, animates via `transform: scaleX()` (never `width`, to stay off the layout-thrashing path) as the reader scrolls a post. Post pages only.

### Page Transitions
- **Style:** Native View Transitions API (`@view-transition { navigation: auto; }`). The outgoing page fades out quickly (0.15s); the incoming page fades and settles up by 6px (0.3s, ease-out-quart) — a quiet arrival rather than the browser's flat default crossfade. Falls back to an instant page swap automatically in browsers without support, and is disabled entirely under `prefers-reduced-motion`.

## 6. Do's and Don'ts

### Do:
- **Do** keep Sage Olive (#687a3f) as the single accent color — header bar, one checked-state, and the reading-progress bar only.
- **Do** keep body copy in the system sans stack; reserve Literata for the wordmark alone.
- **Do** use a 1px dashed hairline (#e0e0e0) for section separation instead of shadows or cards.
- **Do** keep the content column narrow (~40em) so long-form writing stays comfortable to read.
- **Do** let plainness read as a feature: an earnest, understated, first-draft voice, not a polished personal-brand funnel.
- **Do** keep header/nav text white; signal hover with underline or scale, never a dimmer color.
- **Do** animate `transform`/`opacity` for scroll- or hover-linked motion, never `width`/`height` (layout thrash).

### Don't:
- **Don't** introduce hero-metric stat blocks, gradient accents, or glassmorphism — explicit anti-references from PRODUCT.md.
- **Don't** build identical icon-plus-heading card grids for projects, writing, or notes; a plain list row is the established pattern.
- **Don't** add a second saturated brand color; if a new accent is genuinely needed, it must justify displacing Sage Olive's exclusivity, not sit alongside it.
- **Don't** add box-shadows, card elevation, or hover-lift effects — the system is flat by rule, not by omission.
- **Don't** style the wordmark's serif treatment onto body text or headings; Literata is reserved for the logo.
- **Don't** use a colored `border-left`/`border-right` as a side-stripe accent (blockquotes now use italics + indent only, no stripe).
- **Don't** dim header/nav text on hover; any shade darker than white fails 4.5:1 on Sage Olive.
