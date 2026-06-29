// Streams the cited answer from POST /query (SSE over fetch) and renders it
// as a chat conversation, with assistant replies rendered from markdown.
const form = document.getElementById("form");
const questionEl = document.getElementById("question");
const policyEl = document.getElementById("policy_area");
const askBtn = document.getElementById("ask");
const chatInner = document.getElementById("chatInner");
const emptyState = document.getElementById("emptyState");
const chatScroll = document.getElementById("chatScroll");
const themeToggle = document.getElementById("themeToggle");
const themeToggleIcon = themeToggle.querySelector(".theme-toggle-icon");

function applyThemeIcon(theme) {
  // Icon shows the mode a click will switch *to*.
  themeToggleIcon.textContent = theme === "dark" ? "☀️" : "🌙";
}
applyThemeIcon(document.documentElement.getAttribute("data-theme") || "light");

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  applyThemeIcon(next);
});

// Assistant avatar label, set per tenant (brand.initials) on the <body>.
const brandInitials = document.body.dataset.brandInitials || "SCF";

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function addUserMessage(text) {
  if (emptyState) emptyState.remove();
  const msg = document.createElement("div");
  msg.className = "msg user";
  msg.innerHTML =
    `<div class="avatar">You</div>` +
    `<div class="bubble-col"><div class="bubble"></div></div>`;
  msg.querySelector(".bubble").textContent = text;
  chatInner.appendChild(msg);
  scrollToBottom();
}

function addAssistantMessage() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  msg.innerHTML =
    `<div class="avatar"></div>` +
    `<div class="bubble-col">` +
    `<p class="status-line"></p>` +
    `<div class="bubble markdown streaming"></div>` +
    `<details class="sources-toggle"><summary>Sources</summary>` +
    `<div class="sources"></div></details>` +
    `</div>`;
  msg.querySelector(".avatar").textContent = brandInitials;
  chatInner.appendChild(msg);
  scrollToBottom();
  return {
    statusEl: msg.querySelector(".status-line"),
    bubbleEl: msg.querySelector(".bubble"),
    sourcesEl: msg.querySelector(".sources"),
  };
}

// Anthropic emits a citation's metadata right BEFORE the model writes that quoted
// span in the visible text, not after — so we can't place a footnote marker the
// moment the "citation" event arrives. Instead we collect occurrences as they
// stream and, once the full answer is in, locate each quoted span in the final
// text and insert the footnote marker immediately after it.
function citationSearchKey(citedText) {
  return (citedText || "")
    .trim()
    .replace(/^[\w]{1,8}:\s*/, "") // drop a leading "S2: " speaker label, if present
    .replace(/\s+/g, " ")
    .trim();
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const MIN_MATCH_WORDS = 4;

// The model sometimes quotes only a leading portion of the cited passage (drops a
// trailing clause), so retry with progressively shorter trailing word counts before
// giving up — favors the longest match we can actually find in the rendered text.
// Bound how far past the citation's known streamed position we'll search for its quote —
// without this, a coincidental match far ahead can drag the cursor forward and strand
// every citation in between at that same distant spot (the bug that caused footnotes to
// pile up at one location instead of tracking their claims).
const QUOTE_SEARCH_SLACK = 400;

function findQuoteEnd(text, citedText, fromIndex, maxIndex) {
  const key = citationSearchKey(citedText);
  if (!key) return null;
  const searchEnd = typeof maxIndex === "number" ? Math.min(text.length, maxIndex) : text.length;
  if (searchEnd <= fromIndex) return null;
  const window = text.slice(fromIndex, searchEnd);
  const words = key.split(" ").filter(Boolean);
  for (let n = words.length; n >= Math.min(MIN_MATCH_WORDS, words.length); n--) {
    const pattern = words.slice(0, n).map(escapeRegExp).join("\\s+");
    try {
      const re = new RegExp(pattern, "i");
      const m = window.match(re);
      if (m) {
        let end = fromIndex + m.index + m[0].length;
        while (end < text.length && '."?!,;:”\'’'.includes(text[end])) end++;
        return end;
      }
    } catch {
      return null;
    }
  }
  return null;
}

// Paraphrased claims aren't quoted verbatim, so there's no span to match — instead we
// anchor to the streamed character offset of the citation and snap forward to the end of
// the sentence it falls in, so the marker lands after the claim rather than mid-word.
function snapToSentenceEnd(text, from) {
  let end = Math.max(0, Math.min(from, text.length));
  while (end > 0 && /\s/.test(text[end - 1])) end--; // trim a trailing space in the offset
  // Already sitting just after sentence punctuation (+ any closing quote)? Keep it —
  // don't run forward and swallow the next sentence.
  if (/[.?!]["'”’)\]]*$/.test(text.slice(Math.max(0, end - 4), end))) return end;
  // Block ended mid-sentence: extend to the end of the sentence the claim falls in.
  const re = /[.?!]["'”’)\]]*(?=\s|$)/g;
  re.lastIndex = end;
  const m = re.exec(text);
  return m ? m.index + m[0].length : text.length;
}

// Place a footnote marker for each citation. Prefer the precise spot right after a
// verbatim quote of the cited passage; for paraphrased claims (no verbatim span in the
// answer) fall back to the streamed position, snapped to the enclosing sentence's end.
function insertFootnotes(text, occurrences) {
  let result = "";
  let cursor = 0;
  let lastEnd = -1;
  let lastIndex = -1;
  for (const occ of occurrences) {
    const maxIndex =
      typeof occ.pos === "number" ? occ.pos + QUOTE_SEARCH_SLACK : undefined;
    let end = findQuoteEnd(text, occ.cited_text, cursor, maxIndex);
    if (end == null) {
      if (typeof occ.pos !== "number") continue; // no quote and no anchor; skip
      end = snapToSentenceEnd(text, Math.max(cursor, occ.pos));
    }
    if (end === lastEnd && occ.index === lastIndex) continue; // duplicate on same span
    result += text.slice(cursor, end) + `[[${occ.index}]](#cite-${occ.index})`;
    cursor = end;
    lastEnd = end;
    lastIndex = occ.index;
  }
  result += text.slice(cursor);
  return result;
}

function renderSources(sourcesEl, sources) {
  const toggleEl = sourcesEl.closest(".sources-toggle");
  if (!sources || sources.length === 0) {
    if (toggleEl) toggleEl.remove();
    return;
  }
  sourcesEl.innerHTML = "";
  sources.forEach((s) => {
    const src = s.source || {};
    const speakers = (src.speakers || []).join(", ");
    const table = src.table ? `table ${src.table}` : "";
    const loc = [src.session, table, src.date].filter(Boolean).join(" · ");
    const div = document.createElement("div");
    div.className = "source-card";
    div.id = `cite-${s.index}`;
    div.innerHTML = `<div class="loc"></div><div class="quote"></div>`;
    div.querySelector(".loc").textContent =
      `[${s.index}] ${loc} · ${speakers} · ${src.time || ""} ` +
      `(turns ${src.turn_start}-${src.turn_end})`;
    div.querySelector(".quote").textContent = `"${s.cited_text || ""}"`;
    sourcesEl.appendChild(div);
  });
}

async function ask(question, policyArea) {
  askBtn.disabled = true;
  addUserMessage(question);
  const { statusEl, bubbleEl, sourcesEl } = addAssistantMessage();
  statusEl.textContent = "Starting…";

  let raw = "";
  const citationOccurrences = [];

  try {
    const resp = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, policy_area: policyArea || null }),
    });
    if (resp.status === 401) {
      window.location.href = "/";
      return;
    }
    if (!resp.ok || !resp.body) throw new Error(`Request failed (${resp.status})`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete trailing chunk
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "progress") {
          statusEl.textContent = ev.message;
        } else if (ev.type === "token") {
          raw += ev.text;
          bubbleEl.innerHTML = renderMarkdown(raw);
        } else if (ev.type === "citation") {
          citationOccurrences.push({
            index: ev.index,
            cited_text: ev.cited_text,
            pos: ev.pos,
          });
        } else if (ev.type === "done") {
          statusEl.textContent = "";
          bubbleEl.classList.remove("streaming");
          bubbleEl.innerHTML = renderMarkdown(insertFootnotes(raw, citationOccurrences));
          renderSources(sourcesEl, ev.sources);
          scrollToBottom();
        } else if (ev.type === "error") {
          statusEl.innerHTML = `<span class="error-text">Error: ${ev.message}</span>`;
          bubbleEl.classList.remove("streaming");
        }
      }
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="error-text">Error: ${e.message}</span>`;
    bubbleEl.classList.remove("streaming");
  } finally {
    askBtn.disabled = false;
  }
}

function autoGrow() {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 200) + "px";
}
questionEl.addEventListener("input", autoGrow);

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q) return;
  questionEl.value = "";
  autoGrow();
  ask(q, policyEl.value);
});
