(function () {
  "use strict";

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var esc = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };

  var PATTERN_NOTE = {
    "1p": "one hop", "2p": "two-hop chain", "3p": "three-hop chain",
    "2i": "two-way intersection", "ip": "intersect, then project", "pi": "project, then intersect"
  };
  /* ---------------- main results table ---------------- */
  var DS = ["WebQSP", "CWQ", "GrailQA", "GTSQA", "WikiWebQuestions", "Avg."];
  var R = {
    IO: { "3b": [32.21, 20.60, 12.42, 13.72, 11.00, 8.70, 7.83, 7.45, 37.43, 23.00, 20.18, 14.69], "7b": [38.50, 24.20, 24.61, 20.25, 11.70, 8.96, 10.11, 9.16, 45.87, 28.45, 26.16, 18.20], "l8b": [48.14, 31.79, 29.85, 24.30, 16.30, 11.13, 12.95, 11.56, 51.99, 34.09, 31.85, 22.57] },
    CoT: { "3b": [34.96, 22.20, 17.76, 15.00, 11.00, 8.70, 5.73, 4.92, 39.83, 23.41, 21.86, 14.85], "7b": [40.94, 26.98, 28.38, 23.70, 15.40, 12.25, 10.23, 9.15, 47.30, 30.27, 28.45, 20.47], "l8b": [43.99, 30.01, 23.93, 20.18, 15.20, 11.16, 9.06, 8.27, 51.86, 33.87, 28.81, 20.70] },
    SC: { "3b": [50.64, 23.50, 29.31, 14.99, 22.10, 9.81, 13.13, 6.40, 40.50, 25.47, 31.14, 16.03], "7b": [50.58, 27.94, 38.97, 26.08, 23.00, 12.67, 17.76, 9.90, 60.46, 31.19, 38.15, 21.56], "l8b": [61.81, 33.77, 43.39, 25.78, 28.00, 13.80, 20.84, 11.85, 66.71, 35.70, 44.15, 24.18] },
    ReAct: { "3b": [22.63, 18.25, 12.06, 10.63, 21.26, 18.61, 15.47, 14.72, 36.46, 29.56, 21.58, 18.35], "7b": [34.82, 28.42, 16.06, 13.45, 49.19, 50.07, 29.41, 26.52, 49.14, 39.63, 35.72, 31.62], "l8b": [48.42, 40.53, 17.73, 15.00, 27.33, 28.21, 18.87, 17.99, 33.83, 28.73, 29.24, 26.09] },
    ToG: { "7b": [55.22, 43.25, 36.90, 35.70, 62.95, 56.07, 38.04, 34.21, 64.52, 49.07, 51.53, 43.66] },
    PoG: { "7b": [49.23, 36.83, 32.82, 28.57, 57.39, 50.79, 35.02, 32.40, 53.26, 40.14, 45.54, 37.75] },
    CoR: { "7b": [63.27, 57.62, 23.23, 21.32, 58.70, 54.01, 26.63, 24.48, 66.39, 54.00, 47.64, 42.29] },
    "Text-Based Retrieval": { "7b": [59.86, 42.78, 31.61, 28.78, 53.64, 46.60, 37.73, 36.44, 67.22, 50.00, 50.01, 40.92] },
    "Graph Reranker": { "7b": [68.27, 48.89, 37.69, 33.28, 56.28, 48.99, 37.85, 36.65, 70.37, 53.48, 54.09, 44.26] },
    EoG: { "3b": [63.57, 55.22, 38.14, 36.84, 56.17, 55.05, 30.95, 28.76, 41.26, 34.30, 46.02, 42.03], "7b": [70.13, 62.10, 50.58, 49.27, 65.38, 63.01, 48.39, 46.36, 73.28, 60.82, 61.55, 56.31], "l8b": [71.82, 61.51, 50.36, 48.25, 63.87, 62.59, 36.37, 34.67, 70.53, 62.55, 58.59, 53.91] }
  };
  var BACKBONES = [
    { id: "7b", label: "Qwen2.5-7B-Instruct" },
    { id: "3b", label: "Qwen2.5-3B-Instruct" },
    { id: "l8b", label: "LLaMA-3.1-8B-Instruct" }
  ];
  var fmt = function (v) { return v === null ? "–" : v.toFixed(2); };
  function renderTable(bb) {
    var rows = Object.keys(R).filter(function (m) { return R[m][bb]; });
    var best = [];
    for (var c = 0; c < 12; c++) {
      best[c] = Math.max.apply(null, rows.map(function (m) { return R[m][bb][c]; }));
    }
    var h = "<thead><tr><th rowspan=\"2\">Method</th>";
    DS.forEach(function (d) { h += '<th colspan="2">' + d + "</th>"; });
    h += "</tr><tr>";
    DS.forEach(function () { h += "<th>Hit@1</th><th>F1</th>"; });
    h += "</tr></thead><tbody>";
    rows.forEach(function (m) {
      var cls = m === "EoG" ? ' class="eog"' : "";
      h += "<tr" + cls + "><td>" + esc(m) + "</td>";
      R[m][bb].forEach(function (v, c) { h += "<td" + (v === best[c] ? ' class="best"' : "") + ">" + fmt(v) + "</td>"; });
      h += "</tr>";
    });
    $("#main-table").innerHTML = h + "</tbody>";
  }
  var tabsEl = $("#backbone-tabs");
  if (tabsEl) {
    BACKBONES.forEach(function (b, i) {
      var t = document.createElement("button");
      t.type = "button"; t.className = "tab"; t.setAttribute("role", "tab"); t.id = "tab-" + b.id;
      t.setAttribute("aria-selected", i === 0 ? "true" : "false");
      t.textContent = b.label;
      t.addEventListener("click", function () {
        tabsEl.querySelectorAll(".tab").forEach(function (x) { x.setAttribute("aria-selected", "false"); });
        t.setAttribute("aria-selected", "true");
        renderTable(b.id);
      });
      tabsEl.appendChild(t);
    });
    renderTable("7b");
  }

  /* ---------------- cases ---------------- */
  var CASES = window.EOG_CASES || [];
  var NOTES = {
    jersey: "Both roles intersect the countries of the two languages. One Solver rollout follows Jersey's administrative division instead and answers United Kingdom.",
    clary: "Correct rollouts look up Clary Fray's own portrayals; two others stop at the film's cast list and guess Aidan Turner. The Solver splits on this question, so it sits on its capability boundary.",
    hawking: "A three-hop chain: book → author → education record → field of study. Only the book is named in the question."
  };
  var TAGNAME = { think: "think", "kg-query": "kg-query", information: "information", result: "result", answer: "answer" };

  function renderPath(p) {
    // "A <--rel-- Mid1@ --rel--> Ans@"
    var parts = p.split(/(<--[^ ]+--|--[^ ]+-->)/);
    return parts.map(function (t) {
      t = t.trim();
      if (!t) return "";
      if (/^<--|-->$/.test(t)) return '<span class="rel"> ' + esc(t) + " </span>";
      if (t === "Ans@") return '<span class="ph ans">Ans@</span>';
      if (/^Mid\d@$/.test(t)) return '<span class="ph">' + esc(t) + "</span>";
      return '<span class="anc">' + esc(t) + "</span>";
    }).join("");
  }
  function renderInfo(text, uid) {
    var lines = text.split("\n").filter(function (l) { return l.trim(); });
    var LIMIT = 6;
    var html = lines.map(function (l, i) {
      return '<div' + (i >= LIMIT ? ' class="extra" hidden' : "") + ">" + esc(l) + "</div>";
    }).join("");
    if (lines.length > LIMIT) {
      html += '<button type="button" class="more" data-more="' + uid + '">Show all ' + lines.length + " lines</button>";
    }
    return html;
  }
  var uidc = 0;
  function renderSteps(steps, finalWrong) {
    return steps.map(function (s) {
      var id = "i" + (uidc++);
      if (s.t === "think") return '<div class="step think"><div class="tag">&lt;think&gt;</div><div class="body">' + esc(s.x) + "</div></div>";
      if (s.t === "kg-query") return '<div class="step query"><div class="tag">&lt;kg-query&gt;</div><div class="body">' + esc(s.x) + "</div></div>";
      if (s.t === "information") return '<div class="step info"><div class="tag">&lt;information&gt;</div><div class="body" id="' + id + '">' + renderInfo(s.x, id) + "</div></div>";
      var label = TAGNAME[s.t] || s.t;
      var body = s.x;
      if (s.t === "result") { try { body = JSON.parse(s.x).question; } catch (e) { /* keep raw */ } }
      return '<div class="step final' + (finalWrong ? " wrong" : "") + '"><div class="tag">&lt;' + label + "&gt;</div><div class=\"body\">" + esc(body) + "</div></div>";
    }).join("");
  }
  function renderCase(idx) {
    var c = CASES[idx];
    if (!c) return;
    var firstOk = c.rollouts.findIndex(function (r) { return r.ok; });
    var sel = firstOk < 0 ? 0 : firstOk;
    var nOk = c.rollouts.filter(function (r) { return r.ok; }).length;

    var h = '<div class="task">' +
      '<div class="task-row"><div class="task-key">Skeleton</div><div>' + c.path.map(function (p) { return '<div class="path-line">' + renderPath(p) + "</div>"; }).join("") + "</div></div>" +
      '<div class="task-row"><div class="task-key">Gold answer</div><div class="chips">' + c.answers.map(function (a) { return '<span class="chip">' + esc(a) + "</span>"; }).join("") + "</div></div>" +
      "</div>" + '<p class="case-note">' + esc(NOTES[c.id] || "") + "</p>";

    h += '<div class="lanes">';
    h += '<div class="lane p"><div class="lane-head"><div class="lane-title">Proposer</div><div class="lane-sub">Explores from the anchors and writes the question.</div></div>' +
      '<div class="traj">' + renderSteps(c.proposer, false) + "</div></div>";

    h += '<div class="lane s"><div class="lane-head"><div class="lane-title">Solver</div><div class="lane-sub">' + nOk + " of 5 rollouts answer correctly. Click one to view it.</div>" +
      '<div class="rollouts" role="group" aria-label="Solver rollouts">' +
      c.rollouts.map(function (r, i) {
        var pred = r.clean ? r.pred.join(", ") : "no answer";
        return '<button type="button" class="rbtn ' + (r.ok ? "ok" : "bad") + '" data-r="' + i + '" aria-pressed="' + (i === sel) + '" title="Rollout ' + (i + 1) + ": " + esc(pred) + '">' +
          '<span class="mk">' + (r.ok ? "✓" : "✗") + "</span>#" + (i + 1) + ' <span class="pr">' + esc(pred) + "</span></button>";
      }).join("") + "</div></div>" +
      '<div class="traj" id="solver-traj"></div></div>';
    h += "</div>";

    var panel = $("#case-panel");
    panel.innerHTML = h;

    function showRollout(i) {
      var r = c.rollouts[i];
      panel.querySelectorAll(".rbtn").forEach(function (b) { b.setAttribute("aria-pressed", String(+b.getAttribute("data-r") === i)); });
      var t = $("#solver-traj", panel);
      t.innerHTML = r.clean
        ? renderSteps(r.steps, !r.ok)
        : '<div class="no-answer">Rollout ' + (i + 1) + " ends without a well-formed <code>&lt;answer&gt;</code> block, so it scores 0. Its output is not shown.</div>";
    }
    panel.querySelectorAll(".rbtn").forEach(function (b) {
      b.addEventListener("click", function () { showRollout(+b.getAttribute("data-r")); });
    });
    showRollout(sel);
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-more]");
    if (!b) return;
    var box = document.getElementById(b.getAttribute("data-more"));
    box.querySelectorAll(".extra").forEach(function (el) { el.hidden = false; });
    b.remove();
  });
  var caseTabs = $("#case-tabs");
  if (caseTabs && CASES.length) {
    CASES.forEach(function (c, i) {
      var t = document.createElement("button");
      t.type = "button"; t.className = "case-tab"; t.setAttribute("role", "tab"); t.id = "case-tab-" + c.id;
      t.setAttribute("aria-selected", i === 0 ? "true" : "false");
      t.innerHTML = '<span class="pt">' + c.pattern + " (" + esc(PATTERN_NOTE[c.pattern]) + ')</span><br>' + esc(c.question);
      t.addEventListener("click", function () {
        caseTabs.querySelectorAll(".case-tab").forEach(function (x) { x.setAttribute("aria-selected", "false"); });
        t.setAttribute("aria-selected", "true");
        renderCase(i);
      });
      caseTabs.appendChild(t);
    });
    renderCase(0);
  }

  /* ---------------- bibtex copy ---------------- */
  var copyBtn = $("#copy-bib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = $("#bibtex").textContent;
      var done = function () { copyBtn.textContent = "Copied"; setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600); };
      var fallback = function () {
        var range = document.createRange(); range.selectNodeContents($("#bibtex"));
        var s = window.getSelection(); s.removeAllRanges(); s.addRange(range);
        copyBtn.textContent = "Selected, press Ctrl+C";
      };
      try {
        navigator.clipboard.writeText(text).then(done, fallback);
      } catch (e) { fallback(); }
    });
  }

  /* ---------------- lightbox ---------------- */
  var lb = $("#lightbox");
  if (lb && lb.showModal) {
    document.querySelectorAll("img[data-zoom]").forEach(function (img) {
      img.addEventListener("click", function () {
        $("#lightbox-img").src = img.currentSrc || img.src;
        $("#lightbox-img").alt = img.alt;
        lb.showModal();
      });
    });
    $("#lightbox-close").addEventListener("click", function () { lb.close(); });
    lb.addEventListener("click", function (e) { if (e.target === lb) lb.close(); });
  }
})();
