// Sources tab: fetches GET /sources (reconciliation of Qdrant vs. live Google Drive)
// and renders it as a status summary + table. Lazy-loaded on first open, since it
// calls the Drive API — not fetched on every chat page load.
const navChat = document.getElementById("navChat");
const navSources = document.getElementById("navSources");
const chatView = document.getElementById("chatView");
const sourcesView = document.getElementById("sourcesView");
const sourcesRefresh = document.getElementById("sourcesRefresh");
const sourcesSummary = document.getElementById("sourcesSummary");
const sourcesStatus = document.getElementById("sourcesStatus");
const sourcesTableWrap = document.getElementById("sourcesTableWrap");

const STATUS_LABEL = {
  synced: "Synced",
  stale: "Stale",
  missing: "Missing",
  orphaned: "Orphaned",
  local: "Local",
};

let sourcesLoaded = false;

function showView(view) {
  const showSources = view === "sources";
  chatView.hidden = showSources;
  sourcesView.hidden = !showSources;
  navChat.classList.toggle("active", !showSources);
  navChat.setAttribute("aria-pressed", String(!showSources));
  navSources.classList.toggle("active", showSources);
  navSources.setAttribute("aria-pressed", String(showSources));
  if (showSources && !sourcesLoaded) loadSources();
}

navChat.addEventListener("click", () => showView("chat"));
navSources.addEventListener("click", () => showView("sources"));
sourcesRefresh.addEventListener("click", () => loadSources());

function renderSummary(summary) {
  sourcesSummary.innerHTML = "";
  for (const status of Object.keys(STATUS_LABEL)) {
    const count = summary[status] || 0;
    const badge = document.createElement("span");
    badge.className = `status-badge status-${status}`;
    badge.textContent = `${STATUS_LABEL[status]}: ${count}`;
    sourcesSummary.appendChild(badge);
  }
}

function renderTable(rows) {
  if (!rows.length) {
    sourcesTableWrap.innerHTML = "<p class='hint'>No sources found.</p>";
    return;
  }
  const table = document.createElement("table");
  table.className = "sources-table";
  table.innerHTML =
    "<thead><tr>" +
    "<th>Status</th><th>Name</th><th>Session</th><th>Chunks</th><th>Modified</th>" +
    "</tr></thead>";
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");

    const statusTd = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `status-badge status-${row.status}`;
    badge.textContent = STATUS_LABEL[row.status] || row.status;
    statusTd.appendChild(badge);

    const nameTd = document.createElement("td");
    nameTd.textContent = row.name || row.drive_file_id;

    const table_ = row.table ? `table ${row.table}` : "";
    const location = [row.session, table_, row.date].filter(Boolean).join(" · ");
    const locationTd = document.createElement("td");
    locationTd.textContent = location;

    const chunksTd = document.createElement("td");
    chunksTd.textContent = row.chunks;

    const modifiedTd = document.createElement("td");
    modifiedTd.textContent = row.modified_time ? row.modified_time.slice(0, 10) : "";

    tr.append(statusTd, nameTd, locationTd, chunksTd, modifiedTd);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  sourcesTableWrap.innerHTML = "";
  sourcesTableWrap.appendChild(table);
}

function setStatus(text, isError) {
  sourcesStatus.textContent = "";
  if (!text) return;
  const span = document.createElement("span");
  span.className = isError ? "error-text" : "";
  span.textContent = text;
  sourcesStatus.appendChild(span);
}

async function loadSources() {
  sourcesRefresh.disabled = true;
  setStatus("Loading…", false);
  sourcesSummary.innerHTML = "";
  sourcesTableWrap.innerHTML = "";
  try {
    const resp = await fetch("/sources");
    if (resp.status === 401) {
      window.location.href = "/";
      return;
    }
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      throw new Error((body && body.detail) || `Request failed (${resp.status})`);
    }
    const data = await resp.json();
    setStatus("", false);
    renderSummary(data.summary || {});
    renderTable(data.rows || []);
    sourcesLoaded = true;
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  } finally {
    sourcesRefresh.disabled = false;
  }
}
