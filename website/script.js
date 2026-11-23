const nav = document.querySelector(".rail-nav");
const links = nav ? [...nav.querySelectorAll("a")] : [];
const sections = links.map((link) => document.getElementById(link.hash.slice(1)));
const root = document.documentElement;
const rail = document.getElementById("rail");
const railProgress = document.getElementById("rail-progress-bar");
let scrollFrame;

function updateContents() {
  scrollFrame = undefined;
  if (railProgress) {
    const maxScroll = root.scrollHeight - window.innerHeight;
    const ratio = maxScroll > 0 ? Math.min(1, Math.max(0, window.scrollY / maxScroll)) : 0;
    railProgress.style.transform = `scaleY(${ratio.toFixed(4)})`;
  }
  if (!nav) return;

  const readingLine = parseFloat(getComputedStyle(root).scrollPaddingBlockStart) + 12;
  const atBottom = window.scrollY + window.innerHeight >= root.scrollHeight - 2;
  let current = null;
  for (const [index, section] of sections.entries()) {
    if (section.getBoundingClientRect().top <= readingLine) current = links[index];
  }
  if (atBottom) current = links.at(-1);

  const previous = nav.querySelector("[aria-current]");
  if (current === previous) return;
  previous?.removeAttribute("aria-current");
  current?.setAttribute("aria-current", "location");
  if (!current) return;

  const navBounds = nav.getBoundingClientRect();
  const itemBounds = current.getBoundingClientRect();
  nav.scrollBy({
    left: Math.min(0, itemBounds.left - navBounds.left) + Math.max(0, itemBounds.right - navBounds.right),
    top: Math.min(0, itemBounds.top - navBounds.top) + Math.max(0, itemBounds.bottom - navBounds.bottom)
  });
}

function scheduleContentsUpdate() {
  scrollFrame ??= requestAnimationFrame(updateContents);
}

window.addEventListener("scroll", scheduleContentsUpdate, { passive: true });
window.addEventListener("resize", scheduleContentsUpdate);
window.addEventListener("hashchange", scheduleContentsUpdate);
window.addEventListener("load", scheduleContentsUpdate);
scheduleContentsUpdate();

document.querySelectorAll('[role="tablist"]').forEach((tablist) => {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];

  function activate(selected) {
    for (const tab of tabs) {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      const opened = panel.hidden && active;
      panel.hidden = !active;
      if (opened) {
        const reader = panel.matches(".transfer-sheet") ? panel : panel.querySelector(".trace-stage:not([hidden]) .trace-reading, :scope > .trace-reading");
        if (reader) reader.scrollTop = 0;
      }
      if (!active) panel.querySelectorAll("video").forEach((video) => video.pause());
    }
    scheduleContentsUpdate();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let destination;
      switch (event.key) {
        case "ArrowLeft": destination = (index + tabs.length - 1) % tabs.length; break;
        case "ArrowRight": destination = (index + 1) % tabs.length; break;
        case "Home": destination = 0; break;
        case "End": destination = tabs.length - 1; break;
        default: return;
      }
      event.preventDefault();
      activate(tabs[destination]);
      tabs[destination].focus();
    });
  });
});

const player = document.getElementById("rollout-player");
const movie = player.querySelector("video");
const viewport = player.querySelector(".player-viewport");
const scrub = player.querySelector("input[type=range]");
const toggle = player.querySelector("[data-play]");
const zoom = player.querySelector("[data-zoom]");
const clock = player.querySelector("output");
let frameRate = 20;
let drag;
let origin = [50, 50];

function setOrigin(x, y) {
  origin = [x, y].map((value) => Math.min(100, Math.max(0, value)));
  movie.style.transformOrigin = origin.map((v) => v + "%").join(" ");
}
function setZoom(on) {
  viewport.classList.toggle("magnified", on);
  zoom.setAttribute("aria-pressed", String(on));
  setOrigin(50, 50);
}
function refreshPlayer() {
  const duration = Number.isFinite(movie.duration) ? movie.duration : 0;
  scrub.max = duration;
  scrub.value = movie.currentTime;
  const format = (t) => Math.floor(t / 60) + ":" + (t % 60).toFixed(2).padStart(5, "0");
  clock.textContent = format(movie.currentTime) + " / " + format(duration);
  toggle.textContent = movie.paused ? "Play" : "Pause";
}
function togglePlayback() {
  if (movie.paused) movie.play().catch(refreshPlayer);
  else movie.pause();
}
function stepFrame(direction) {
  if (!Number.isFinite(movie.duration)) return;
  movie.pause();
  const frame = Math.round(movie.currentTime * frameRate);
  movie.currentTime = Math.min(Math.max(0, movie.duration - 0.5 / frameRate), Math.max(0, (frame + direction) / frameRate));
}
const roundsByClip = new Map([...document.querySelectorAll(".trace-stage[data-stage-key]")].map((stage) => [stage.dataset.stageKey, stage]));
function updateReadingLink(link, episode) {
  const stage = roundsByClip.get(episode);
  link.hidden = !stage;
  if (stage) link.setAttribute("href", "#" + stage.id);
  else link.removeAttribute("href");
}
function revealReading(scroll) {
  const stage = document.getElementById(location.hash.slice(1));
  if (!stage?.classList.contains("trace-stage")) return;
  const trace = stage.closest(".trace-case");
  document.getElementById(trace.getAttribute("aria-labelledby")).click();
  document.getElementById(stage.getAttribute("aria-labelledby")).click();
  if (scroll) {
    stage.tabIndex = -1;
    stage.focus({preventScroll: true});
    stage.scrollIntoView({block: "start"});
  }
}
document.addEventListener("click", (event) => {
  const link = event.target.closest?.(".card-trace-link, [data-read-current]");
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  const destination = link.hash;
  if (player.open) player.close();
  if (location.hash === destination) revealReading(true);
  else location.hash = destination;
});
window.addEventListener("hashchange", () => revealReading(true));
window.addEventListener("load", () => revealReading(true));

let playingCard;
function openPlayer(card, choice) {
  const roundBar = player.querySelector(".player-rounds");
  if (playingCard !== card) {
    roundBar.replaceChildren();
    const options = [...card.querySelectorAll("[data-clip]")];
    roundBar.hidden = options.length < 2;
    if (options.length > 1) {
      for (const option of options) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = option.dataset.round;
        button.dataset.key = option.dataset.key;
        button.setAttribute("aria-label", option.getAttribute("aria-label"));
        button.addEventListener("click", () => {
          if (button.getAttribute("aria-pressed") === "true") return;
          option.click();
          openPlayer(card, option);
        });
        roundBar.append(button);
      }
    }
    playingCard = card;
  }
  for (const button of roundBar.children) {
    button.setAttribute("aria-pressed", String(button.dataset.key === choice.dataset.key));
  }
  const source = choice.dataset;
  player.querySelector("h3").textContent = card.querySelector("h4").textContent;
  updateReadingLink(player.querySelector("[data-read-current]"), source.key);
  const revision = source.round ? " · " + source.round : "";
  player.querySelector(".player-meta").textContent = card.querySelector(".rollout-context").textContent + revision + " · " + source.outcome + " · " + source.size;
  player.querySelector("[data-original]").href = source.clip;
  viewport.style.setProperty("--ratio", source.ratio);
  player.style.setProperty("--ratio", source.ratio);
  frameRate = Number(source.fps);
  scrub.step = 1 / frameRate;
  setZoom(false);
  movie.poster = source.poster;
  movie.src = source.clip;
  movie.load();
  if (!player.open) player.showModal();
  movie.play().catch(refreshPlayer);
}
for (const link of document.querySelectorAll(".feature-play")) {
  link.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || !player.showModal) return;
    const choice = document.querySelector('.rollout-card [data-key="' + link.dataset.stageKey + '"]');
    if (!choice) return;
    event.preventDefault();
    choice.click();
    openPlayer(choice.closest(".rollout-card"), choice);
  });
}

for (const link of document.querySelectorAll(".transfer-play")) {
  link.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || !player.showModal) return;
    event.preventDefault();
    openPlayer(link.closest(".paired-film"), link);
  });
}

for (const card of document.querySelectorAll(".rollout-card")) {
  const open = card.querySelector(".rollout-open");
  const choices = [...card.querySelectorAll("[data-clip]")];
  let active = choices[0];

  const badgeText = card.querySelector(".success-round")?.textContent?.trim();
  if (badgeText && !open.querySelector(".rollout-badge")) {
    const badge = document.createElement("span");
    badge.className = "rollout-badge";
    badge.textContent = badgeText;
    open.append(badge);
  }

  const outcomeEl = card.querySelector(".round-outcome");
  const foot = document.createElement("div");
  foot.className = "rollout-foot";
  if (outcomeEl) foot.append(outcomeEl);

  const reading = document.createElement("a");
  reading.className = "card-trace-link";
  reading.textContent = "Read round ↗";
  foot.append(reading);
  card.append(foot);

  updateReadingLink(reading, active.dataset.key);
  for (const choice of choices) {
    choice.addEventListener("click", () => {
      active = choice;
      updateReadingLink(reading, active.dataset.key);
      for (const other of choices) other.setAttribute("aria-pressed", String(other === choice));
      open.href = choice.dataset.clip;
      open.querySelector("img").src = choice.dataset.poster;
      if (outcomeEl) outcomeEl.textContent = choice.dataset.outcome;
    });
  }
  open.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || !player.showModal) return;
    event.preventDefault();
    openPlayer(card, active);
  });
}
const gallery = document.getElementById("demo-gallery");
const cards = [...gallery.querySelectorAll(".rollout-card")];
const collections = [...gallery.querySelectorAll('[name="gallery-view"]')];
const search = gallery.querySelector('input[type="search"]');
const facet = gallery.querySelector("select");
const galleryCount = gallery.querySelector(".gallery-count");
const grid = gallery.querySelector(".rollout-grid");
const benchGroups = [["libero", "LIBERO-PRO"], ["robo", "RoboDojo"]];
const commonFacets = [["", "All cases"], ["recovered", "Failure → success"], ["unresolved", "No success shown"]];
const facets = {
  recovered: [["", "Both benchmarks"], ["libero", "LIBERO-PRO"], ["robo", "RoboDojo"]],
  libero: [...commonFacets, ["development", "Development series"], ["heldout", "Held-out episodes"], ["long", "Long suite"], ["goal", "Goal suite"], ["spatial", "Spatial suite"], ["object", "Object suite"]],
  robo: commonFacets
};
function collection() {
  return collections.find((item) => item.checked).value;
}
function matchesFilter(card, value) {
  return !value || value === card.dataset.bench || card.dataset.filters.split(" ").includes(value);
}

function updateLaneUI(lane) {
  const track = lane.querySelector(".rollout-lane-track");
  const shell = lane.querySelector(".rollout-lane-shell");
  const prev = lane.querySelector('[data-lane-dir="-1"]');
  const next = lane.querySelector('[data-lane-dir="1"]');
  const thumb = lane.querySelector(".lane-thumb");
  if (!track || !shell) return;

  const maxScroll = Math.max(0, track.scrollWidth - track.clientWidth);
  const left = track.scrollLeft;
  shell.classList.toggle("has-left", left > 8);
  shell.classList.toggle("has-right", maxScroll - left > 8);
  if (prev) prev.disabled = left <= 4;
  if (next) next.disabled = maxScroll - left <= 4;

  if (thumb) {
    const visibleRatio = track.scrollWidth > 0 ? track.clientWidth / track.scrollWidth : 1;
    const thumbWidthPct = Math.max(12, Math.min(100, visibleRatio * 100));
    const progress = maxScroll > 0 ? left / maxScroll : 0;
    const travelPct = 100 - thumbWidthPct;
    thumb.style.inlineSize = `${thumbWidthPct}%`;
    thumb.style.transform = `translateX(${(progress * (travelPct / thumbWidthPct) * 100).toFixed(2)}%)`;
  }
}

function bindLaneInteractions(lane) {
  const track = lane.querySelector(".rollout-lane-track");
  if (!track) return;

  const stepSize = () => Math.max(280, Math.round(track.clientWidth * 0.72));
  lane.querySelector('[data-lane-dir="-1"]')?.addEventListener("click", () => {
    track.scrollBy({ left: -stepSize(), behavior: "smooth" });
  });
  lane.querySelector('[data-lane-dir="1"]')?.addEventListener("click", () => {
    track.scrollBy({ left: stepSize(), behavior: "smooth" });
  });

  track.addEventListener("scroll", () => requestAnimationFrame(() => updateLaneUI(lane)), { passive: true });
  track.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      track.scrollBy({ left: -stepSize(), behavior: "smooth" });
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      track.scrollBy({ left: stepSize(), behavior: "smooth" });
    }
  });

  let dragStart;
  track.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0 || event.target.closest("button, a")) return;
    dragStart = { x: event.clientX, left: track.scrollLeft };
    track.setPointerCapture(event.pointerId);
    track.classList.add("is-dragging");
  });
  track.addEventListener("pointermove", (event) => {
    if (dragStart) track.scrollLeft = dragStart.left - (event.clientX - dragStart.x);
  });
  track.addEventListener("lostpointercapture", () => {
    dragStart = undefined;
    track.classList.remove("is-dragging");
  });

  requestAnimationFrame(() => updateLaneUI(lane));
}

function displayGallery() {
  const words = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = cards.filter((card) => matchesFilter(card, collection()) && matchesFilter(card, facet.value) && words.every((word) => card.dataset.search.includes(word)));
  const shown = new Set(matches);
  for (const card of cards) card.hidden = !shown.has(card);

  // Move cards temporarily to fragment or detach lanes cleanly
  const staging = document.createDocumentFragment();
  for (const card of cards) staging.append(card);
  grid.replaceChildren();

  for (const [bench, label] of benchGroups) {
    const inGroup = matches.filter((card) => card.dataset.bench === bench);
    if (!inGroup.length) continue;

    const lane = document.createElement("section");
    lane.className = "rollout-lane";
    lane.setAttribute("aria-label", `${label} rollouts`);

    const rows = inGroup.length > 8 ? 2 : 1;
    lane.innerHTML = `
      <div class="rollout-lane-head">
        <div class="rollout-lane-title">
          <h3>${label}</h3>
          <span class="lane-count">${inGroup.length} task${inGroup.length === 1 ? "" : "s"}</span>
          <span class="lane-hint">Scroll or drag &rarr;</span>
        </div>
        <div class="lane-arrows">
          <button class="lane-btn" type="button" data-lane-dir="-1" aria-label="Scroll ${label} left">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>
          </button>
          <button class="lane-btn" type="button" data-lane-dir="1" aria-label="Scroll ${label} right">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
        </div>
      </div>
      <div class="rollout-lane-shell">
        <div class="rollout-lane-track" data-rows="${rows}" tabindex="0" role="region" aria-label="${label} scrollable rollout list"></div>
        <div class="lane-progress" aria-hidden="true"><span class="lane-thumb"></span></div>
      </div>
    `;

    const track = lane.querySelector(".rollout-lane-track");
    for (const card of inGroup) track.append(card);
    grid.append(lane);
    bindLaneInteractions(lane);
  }

  if (!matches.length) {
    const empty = document.createElement("p");
    empty.className = "gallery-empty";
    empty.textContent = "No matching tasks for this filter.";
    grid.append(empty);
  }

  // Keep unmatched hidden cards inside gallery so future queries find them
  const hiddenPool = document.createElement("div");
  hiddenPool.hidden = true;
  hiddenPool.append(staging);
  grid.append(hiddenPool);

  galleryCount.textContent = matches.length ? `${matches.length} task${matches.length === 1 ? "" : "s"}` : "No matching tasks";
  scheduleContentsUpdate();
}

window.addEventListener("resize", () => {
  grid.querySelectorAll(".rollout-lane").forEach(updateLaneUI);
});

document.querySelector("[data-show-recovered]").addEventListener("click", (event) => {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  collections.find((item) => item.value === "recovered").checked = true;
  search.value = "";
  changeCollection();
});

function resetPage() {
  displayGallery();
}
function changeCollection() {
  facet.replaceChildren(...facets[collection()].map(([value, label]) => new Option(label, value)));
  resetPage();
}
for (const item of collections) item.addEventListener("change", changeCollection);
search.addEventListener("input", resetPage);
facet.addEventListener("change", resetPage);
changeCollection();

toggle.addEventListener("click", togglePlayback);
for (const key of ["play", "pause", "timeupdate", "loadedmetadata", "emptied"]) movie.addEventListener(key, refreshPlayer);
scrub.addEventListener("input", () => { movie.pause(); movie.currentTime = Number(scrub.value); });
for (const step of player.querySelectorAll("[data-step]")) step.addEventListener("click", () => stepFrame(Number(step.dataset.step)));
zoom.addEventListener("click", () => setZoom(!viewport.classList.contains("magnified")));
viewport.addEventListener("pointerdown", (event) => {
  if (!viewport.classList.contains("magnified")) return;
  drag = [event.clientX, event.clientY, ...origin];
  viewport.setPointerCapture(event.pointerId);
});
viewport.addEventListener("pointermove", (event) => {
  if (!drag) return;
  setOrigin(drag[2] - 100 * (event.clientX - drag[0]) / viewport.clientWidth, drag[3] - 100 * (event.clientY - drag[1]) / viewport.clientHeight);
});
viewport.addEventListener("lostpointercapture", () => { drag = undefined; });
viewport.addEventListener("keydown", (event) => {
  const offset = { ArrowLeft: [-5,0], ArrowRight: [5,0], ArrowUp: [0,-5], ArrowDown: [0,5] }[event.key];
  if (offset && viewport.classList.contains("magnified")) {
    event.preventDefault(); setOrigin(origin[0] + offset[0], origin[1] + offset[1]);
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault(); stepFrame(event.key === "ArrowLeft" ? -1 : 1);
  } else if (event.key === " ") { event.preventDefault(); togglePlayback(); }
});
player.querySelector("[data-close]").addEventListener("click", () => player.close());
player.addEventListener("click", (event) => {
  if (event.target !== player) return;
  const box = player.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) player.close();
});
player.addEventListener("close", () => { movie.pause(); movie.removeAttribute("src"); movie.removeAttribute("poster"); movie.load(); });
for (const video of document.querySelectorAll(".trace-stage video")) {
  const path = video.getAttribute("src") || video.querySelector("source")?.getAttribute("src");
  const choice = [...document.querySelectorAll(".rollout-card [data-clip]")].find((el) => el.dataset.clip === path);
  if (!choice) continue;
  const link = document.createElement("a");
  link.href = path;
  link.className = "trace-player-link";
  link.textContent = "Inspect full size ↗";
  video.closest("figure").append(link);
  link.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || !player.showModal) return;
    event.preventDefault();
    const time = video.currentTime;
    const paused = video.paused;
    video.pause();
    openPlayer(choice.closest(".rollout-card"), choice);
    movie.addEventListener("loadedmetadata", () => {
      movie.currentTime = time;
      if (paused) movie.pause();
    }, {once: true});
  });
}

for (const stage of document.querySelectorAll(".trace-stage")) {
  const note = stage.querySelector(".frame-note");
  const buttons = [...stage.querySelectorAll(".frame-options button")];
  const video = stage.querySelector("video");
  const still = stage.querySelector(".trace-still");
  const initialNote = note.textContent;
  let requestedTime;

  const roundTab = document.getElementById(stage.getAttribute("aria-labelledby"));
  if (roundTab && !roundTab.dataset.status) {
    const label = roundTab.textContent.toLowerCase();
    if (label.includes("success") || label.includes("6 / 6") || label.includes("10 / 10") || label.includes("stacked")) roundTab.dataset.status = "pass";
    else if (label.includes("replay") || label.includes("φ")) roundTab.dataset.status = "replay";
    else roundTab.dataset.status = "fail";
  }

  const reading = stage.querySelector(".trace-reading");
  if (reading && !reading.dataset.enhanced) {
    reading.dataset.enhanced = "1";
    const blocks = [...reading.querySelectorAll(":scope > .trace-block")];
    const roleOrder = { signal: 0, mechanism: 1, topology: 2 };
    let detectedLayer = "";
    let detectedTransition = "";

    const classifyBlock = (block) => {
      const h5 = block.querySelector("h5");
      const rawTitle = h5?.textContent.trim() || "";
      const lower = rawTitle.toLowerCase();
      const excerptHead = (block.querySelector(".excerpt > p")?.textContent || "").toLowerCase();

      const transMatch = rawTitle.match(/(H[₀-₉0-9]+\s*→\s*H[₀-₉0-9]+|Φ[₀-₉0-9]+\s*→\s*Φ[₀-₉0-9]+)/);
      if (transMatch && !detectedTransition) detectedTransition = transMatch[1];

      let role = "signal";
      let badge = "01 · Observation";
      if (/topology|graph|route|reading|required states/.test(lower)) {
        role = "topology";
        badge = /original reading/.test(lower) ? "03 · Prior Reading" : "03 · Topology & State Change";
      } else if (/harness|context|workflow|control-loop|control loop|running loop|baseline|starting|placement/.test(lower)) {
        role = "mechanism";
        if (/baseline|starting/.test(lower)) {
          detectedLayer = detectedLayer || "Baseline H₀";
          badge = "02 · Baseline Harness";
        } else if (/workflow/.test(lower) || excerptHead.includes("workflow")) {
          detectedLayer = "Workflow W";
          badge = "02 · Mechanism (Workflow W)";
        } else if (/loop/.test(lower) || excerptHead.includes("loop")) {
          detectedLayer = "Control Loop L";
          badge = "02 · Mechanism (Control Loop L)";
        } else if (/context|placement/.test(lower) || excerptHead.includes("context")) {
          detectedLayer = "Context C";
          badge = "02 · Mechanism (Context C)";
        } else {
          detectedLayer = detectedLayer || "(C, W, L) Edit";
          badge = "02 · Harness Mechanism";
        }
      } else {
        if (/stalled/.test(lower)) badge = "01 · Physical Bottleneck";
        else if (/completion/.test(lower)) badge = "01 · Verified Completion";
        else if (/outcome/.test(lower)) badge = "01 · Observed Outcome";
      }

      block.dataset.role = role;
      if (h5) {
        const tag = document.createElement("span");
        tag.className = "trace-step-badge";
        tag.textContent = badge;
        h5.append(tag);
      }
      return role;
    };

    const sorted = blocks
      .map((b, idx) => ({ block: b, role: classifyBlock(b), idx }))
      .sort((a, b) => (roleOrder[a.role] - roleOrder[b.role]) || (a.idx - b.idx));

    const flow = document.createElement("div");
    flow.className = "trace-flow";
    for (const item of sorted) flow.append(item.block);

    const title = reading.querySelector("h4");
    title.id = stage.id + "-reading-title";
    reading.setAttribute("aria-labelledby", title.id);
    reading.tabIndex = 0;

    const jumps = document.createElement("nav");
    jumps.className = "trace-jumps";
    jumps.setAttribute("aria-label", "Jump within this revision");
    for (const [role, label] of [["signal", "Observation"], ["mechanism", "Harness"], ["topology", "Topology"]]) {
      const block = sorted.find((item) => item.role === role)?.block;
      if (!block) continue;
      block.tabIndex = -1;
      const jump = document.createElement("button");
      jump.type = "button";
      jump.textContent = label;
      jump.addEventListener("click", () => {
        block.focus({ preventScroll: true });
        const offset = block.getBoundingClientRect().top - reading.getBoundingClientRect().top;
        reading.scrollTop += offset - jumps.offsetHeight - 12;
      });
      jumps.append(jump);
    }
    reading.append(jumps, flow);

    const heading = reading.querySelector("h4");
    if (heading && roundTab) {
      const bar = document.createElement("div");
      bar.className = "trace-causal-bar";

      const roundTitle = roundTab.querySelector("b")?.textContent.trim() || "";
      const roundState = roundTab.querySelector("span")?.textContent.trim() || "";
      const statusChip = document.createElement("span");
      statusChip.className = "trace-causal-chip";
      statusChip.dataset.chip = "status";
      statusChip.dataset.status = roundTab.dataset.status;
      statusChip.textContent = roundState ? `${roundTitle} · ${roundState}` : roundTitle;
      bar.append(statusChip);

      if (detectedLayer || roundTab.dataset.status === "replay") {
        const layerChip = document.createElement("span");
        layerChip.className = "trace-causal-chip";
        layerChip.dataset.chip = "layer";
        layerChip.textContent = roundTab.dataset.status === "replay"
          ? "Retrospective Replay (Φ₁ → Φ₂)"
          : `Layer: ${detectedLayer}`;
        bar.append(layerChip);
      }

      if (detectedTransition) {
        const transChip = document.createElement("span");
        transChip.className = "trace-causal-chip";
        transChip.textContent = `Edit: ${detectedTransition}`;
        bar.append(transChip);
      }

      heading.after(bar);
    }
  }

  function selectFrame(selected) {
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button === selected)));
    note.textContent = selected?.dataset.caption ?? initialNote;
  }

  for (const button of buttons) {
    button.addEventListener("click", () => {
      if (video) {
        video.pause();
        requestedTime = Number(button.dataset.seek);
        if (video.readyState >= 1) video.currentTime = requestedTime;
        else if (video.networkState !== 2) video.load();
      } else {
        still.src = button.dataset.observation;
        still.alt = button.dataset.caption;
        still.closest("a").href = button.dataset.observation;
      }
      selectFrame(button);
    });
  }
  video?.addEventListener("loadedmetadata", () => {
    if (requestedTime !== undefined) video.currentTime = requestedTime;
  });
  video?.addEventListener("timeupdate", () => {
    const atFrame = buttons.find((button) => Math.abs(Number(button.dataset.seek) - video.currentTime) < 0.035);
    selectFrame(atFrame);
  });
}

document.addEventListener("play", (event) => {
  if (event.target.tagName === "VIDEO") {
    document.querySelectorAll("video").forEach((other) => {
      if (other !== event.target && !other.paused) other.pause();
    });
  }
}, true);

const reprojData = {
  phi1: {
    graspFrames: "[70, 140]",
    progress: "0.50",
    changed: false,
    note:
      "Under initial topology Φ₁, supported deposit is treated as required. Failed rollout H₁ receives acquisition credit across frames [70, 140], giving a terminal progress of 0.50."
  },
  phi2: {
    graspFrames: "[70, 90]",
    progress: "0.33",
    changed: true,
    note:
      "H₂ first separates the diagnostic lift from lateral motion. Its tilted success informs Φ₂, where supported deposit becomes optional. Re-reading the exact same H₁ trace narrows the grasp interval to [70, 90] and updates its progress to 0.33."
  }
};

function renderReproj(mode) {
  const container = document.getElementById("reproj-display");
  const d = reprojData[mode];
  container.innerHTML = `
      <div class="metric-pair">
        <div class="metric-chip ${d.changed ? "changed" : ""}">
          <div class="label">H₁ grasp frames</div>
          <div class="val">${d.graspFrames}</div>
        </div>
        <div class="metric-chip ${d.changed ? "changed" : ""}">
          <div class="label">H₁ Terminal Progress</div>
          <div class="val">${d.progress}</div>
        </div>
      </div>
    <p class="state-note">${d.note}</p>
  `;
}

const reprojectionButtons = [...document.querySelectorAll("[data-reproj]")];
for (const button of reprojectionButtons) {
  button.addEventListener("click", () => {
    for (const option of reprojectionButtons) option.setAttribute("aria-pressed", String(option === button));
    renderReproj(button.dataset.reproj);
    scheduleContentsUpdate();
  });
}
if (reprojectionButtons.length) renderReproj(reprojectionButtons[0].dataset.reproj);

const copyButton = document.getElementById("copy-bibtex");
const copyStatus = document.getElementById("copy-status");
const citation = document.querySelector("#bibtex-code code");
let copyTimer;

copyButton.addEventListener("click", async () => {
  clearTimeout(copyTimer);
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = "Copied";
    copyStatus.textContent = "BibTeX copied to clipboard";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = "BibTeX selected";
    copyStatus.textContent = "BibTeX selected. Use your keyboard to copy it.";
  }
  copyTimer = setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
    copyStatus.textContent = "";
  }, 1800);
});

const lightbox = document.getElementById("lightbox");
const expandedImage = document.getElementById("lightbox-img");

document.querySelectorAll(".figure-link").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    event.preventDefault();
    expandedImage.src = link.href;
    expandedImage.alt = link.querySelector("img").alt;
    lightbox.showModal();
  });
});

lightbox.querySelector("button").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

for (const formula of document.querySelectorAll("[data-math]")) {
  katex.render(formula.textContent.trim(), formula, {
    displayMode: formula.dataset.math === "display"
  });
}
