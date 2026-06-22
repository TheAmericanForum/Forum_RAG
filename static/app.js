// Streams the cited answer from POST /query (SSE over fetch) and renders it
// as a chat conversation, with assistant replies rendered from markdown.
const form = document.getElementById("form");
const questionEl = document.getElementById("question");
const policyEl = document.getElementById("policy_area");
const askBtn = document.getElementById("ask");
const chatInner = document.getElementById("chatInner");
const emptyState = document.getElementById("emptyState");
const chatScroll = document.getElementById("chatScroll");

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
    `<div class="avatar">SC</div>` +
    `<div class="bubble-col">` +
    `<p class="status-line"></p>` +
    `<div class="bubble markdown streaming"></div>` +
    `<div class="sources"></div>` +
    `</div>`;
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
function findQuoteEnd(text, citedText, fromIndex) {
  const key = citationSearchKey(citedText);
  if (!key) return null;
  const words = key.split(" ").filter(Boolean);
  for (let n = words.length; n >= Math.min(MIN_MATCH_WORDS, words.length); n--) {
    const pattern = words.slice(0, n).map(escapeRegExp).join("\\s+");
    try {
      const re = new RegExp(pattern, "i");
      const m = text.slice(fromIndex).match(re);
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

function insertFootnotes(text, occurrences) {
  let result = "";
  let cursor = 0;
  for (const occ of occurrences) {
    const end = findQuoteEnd(text, occ.cited_text, cursor);
    if (end == null) continue; // couldn't locate the quote verbatim; skip rather than misplace
    result += text.slice(cursor, end) + `[[${occ.index}]](#cite-${occ.index})`;
    cursor = end;
  }
  result += text.slice(cursor);
  return result;
}

function renderSources(sourcesEl, sources) {
  if (!sources || sources.length === 0) return;
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
          scrollToBottom();
        } else if (ev.type === "citation") {
          citationOccurrences.push({ index: ev.index, cited_text: ev.cited_text });
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
