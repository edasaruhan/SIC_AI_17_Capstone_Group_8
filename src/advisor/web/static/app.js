/* kısaliste: one page, no framework. Every model or site text reaches the DOM through
   textContent, never innerHTML, so nothing a website or an assistant wrote can run here. */
"use strict";

const app = document.getElementById("app");
const state = {
  view: "analyze",
  stage: "start", // start | profile | running | result
  value: "",
  profile: null,
  queries: { discovery: [], named: [] },
  options: { second: false, auditAi: true, reps: 2 },
  estimate: null,
  job: null,
  steps: [],
  result: null,
  busy: false,
  error: null,
  audit: { text: "", rivals: "", ai: false, result: null, busy: false, error: null },
  evidence: null,
};

const STEP_ORDER = ["plan", "search", "discover", "interrogate", "analyse", "describe", "report"];
const STEP_TEXT = {
  plan: "Sorular hazırlanıyor",
  search: "Google'da aranıyor",
  discover: "Rakip markalar bulunuyor",
  interrogate: "Yapay zekâ asistanlarına soruluyor",
  analyse: "Yanıtlar ölçülüyor, teşhis konuyor",
  describe: "Ürün açıklaması inceleniyor",
  report: "Rapor yazılıyor",
};
const VERDICT_TEXT = {
  strong: "İki asistanda da kazandırıyor",
  mixed: "Asistana bağlı",
  weak: "Belirgin kazanç yok",
  risk: "Risk",
};

/* ---------- helpers ---------- */

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "value") node.value = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, body, method = "POST") {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `İstek başarısız oldu (${response.status}).`);
  }
  return data;
}

const pct = (value) => (value === null || value === undefined ? "–" : `%${Math.round(value * 100)}`);
const fold = (text) => text.toLocaleLowerCase("tr");

function marked(text, own, rivals) {
  // Split the text around brand names; own spellings get the highlighter, rivals a grey mark.
  const names = [
    ...own.filter(Boolean).map((n) => ({ n, cls: "own" })),
    ...rivals.filter(Boolean).map((n) => ({ n, cls: "rival" })),
  ].sort((a, b) => b.n.length - a.n.length);
  if (!names.length) return [text];
  const lower = fold(text);
  const out = [];
  let i = 0;
  while (i < text.length) {
    let hit = null;
    for (const item of names) {
      const at = lower.indexOf(fold(item.n), i);
      if (at !== -1 && (!hit || at < hit.at)) hit = { at, ...item };
    }
    if (!hit) {
      out.push(text.slice(i));
      break;
    }
    if (hit.at > i) out.push(text.slice(i, hit.at));
    out.push(h("mark", { class: hit.cls }, text.slice(hit.at, hit.at + hit.n.length)));
    i = hit.at + hit.n.length;
  }
  return out;
}

function inline(text, own = [], rivals = []) {
  // **bold** becomes <strong> and <br> a line break; any other markdown mark is dropped.
  // Still text nodes only.
  return text.split(/<br\s*\/?>/i).map((chunk, index) => [
    index ? h("br", {}) : null,
    chunk.split(/(\*\*[^*]+\*\*)/g).map((part) =>
      part.length > 4 && part.startsWith("**") && part.endsWith("**")
        ? h("strong", {}, marked(part.slice(2, -2), own, rivals))
        : marked(part.replace(/\*\*|__|`/g, ""), own, rivals)
    ),
  ]);
}

const RULE_ROW = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/;

function cells(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|")) row = row.slice(0, -1);
  return row.split("|").map((cell) => cell.trim());
}

function mdTable(rows, own, rivals) {
  // Same reading as the PDF (advisor/web/pdf.py): a rule row under the first row makes it the header.
  const parsed = rows.map((row) => ({ rule: RULE_ROW.test(row), cells: cells(row) }));
  const hasHead = parsed.length > 1 && parsed[1].rule;
  const body = parsed.filter((row) => !row.rule).map((row) => row.cells);
  if (!body.length) return null;
  const width = Math.max(...body.map((row) => row.length));
  const row = (values, tag) =>
    h("tr", {}, [...values, ...Array(width - values.length).fill("")].map((value) => h(tag, {}, inline(value, own, rivals))));
  return h(
    "div",
    { class: "table-wrap" },
    h("table", { class: "md-table" }, hasHead ? h("thead", {}, row(body[0], "th")) : null, h("tbody", {}, (hasHead ? body.slice(1) : body).map((values) => row(values, "td"))))
  );
}

function prose(text, own = [], rivals = []) {
  // Headings, bullets and numbered items from an assistant's markdown, as plain elements.
  // Quote marks ("> ") carry no meaning in an answer card; the text under them does.
  const lines = text.replace(/\r/g, "").split("\n").map((raw) => raw.trim().replace(/^>\s?/, "").trim());
  const nextFilled = (i) => {
    while (i < lines.length && !lines[i]) i++;
    return lines[i] || "";
  };
  const blocks = [];
  let list = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line) {
      list = null;
      continue;
    }
    if (line.startsWith("|")) {
      // Models often leave a blank line between a table's rows; it is still one table.
      list = null;
      const rows = [];
      while (i < lines.length && (lines[i].startsWith("|") || (!lines[i] && nextFilled(i).startsWith("|")))) {
        if (lines[i]) rows.push(lines[i]);
        i++;
      }
      i--;
      blocks.push(mdTable(rows, own, rivals));
      continue;
    }
    if (/^[-*_]{3,}$/.test(line)) {
      list = null;
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^(?:[*\-•]|\d+[.)])\s+(.*)$/);
    if (heading) {
      list = null;
      blocks.push(h("p", { class: "md-h" }, inline(heading[1], own, rivals)));
    } else if (bullet) {
      if (!list) {
        list = h("ul", { class: "md-list" });
        blocks.push(list);
      }
      list.append(h("li", {}, inline(bullet[1], own, rivals)));
    } else {
      list = null;
      blocks.push(h("p", {}, inline(line, own, rivals)));
    }
  }
  return blocks;
}

function answerBody(text, own, rivals) {
  // Long answers open folded; nothing is cut, the rest is one click away.
  const body = h("div", { class: "a" }, prose(text, own, rivals));
  if (text.length < 900) return body;
  body.classList.add("folded");
  const more = h("button", { class: "linkish more", type: "button", "aria-expanded": "false", text: "Yanıtın tamamını göster" });
  more.onclick = () => {
    const open = !body.classList.toggle("folded");
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "Kısalt" : "Yanıtın tamamını göster";
  };
  return [body, more];
}

function render() {
  document.querySelectorAll(".nav button").forEach((b) => {
    b.toggleAttribute("aria-current", b.dataset.view === state.view);
    if (b.dataset.view === state.view) b.setAttribute("aria-current", "page");
  });
  app.replaceChildren();
  if (state.view === "audit") app.append(auditView());
  else if (state.view === "method") app.append(methodView());
  else app.append(analyzeView());
}

function go(view) {
  state.view = view;
  render();
  app.focus({ preventScroll: true });
  window.scrollTo({ top: 0 });
}

document.querySelectorAll("[data-view]").forEach((el) =>
  el.addEventListener("click", (event) => {
    event.preventDefault();
    go(el.dataset.view);
  })
);

/* ---------- analyze: start ---------- */

function steps(current) {
  const labels = ["Marka", "Sorular", "Analiz", "Sonuç"];
  const index = { profile: 1, running: 2, result: 3 }[current] ?? 0;
  return h(
    "ol",
    { class: "steps", "aria-label": "Adımlar" },
    labels.map((label, i) =>
      h("li", { "data-state": i < index ? "done" : i === index ? "now" : "todo" }, h("span", {}, i + 1), label)
    )
  );
}

function analyzeView() {
  if (state.stage === "profile") return profileView();
  if (state.stage === "running") return runningView();
  if (state.stage === "result") return resultView();
  return startView();
}

function startView() {
  const input = h("input", {
    class: "input",
    id: "brand-input",
    type: "text",
    autocomplete: "off",
    placeholder: "garantibbva.com.tr ya da Garanti BBVA",
    value: state.value,
    "aria-label": "Web sitesi ya da marka adı",
    oninput: (e) => (state.value = e.target.value),
  });
  const submit = async (event) => {
    event.preventDefault();
    if (state.value.trim().length < 2 || state.busy) return;
    state.busy = true;
    state.error = null;
    render();
    try {
      const data = await api("/api/profile", { value: state.value.trim() });
      state.profile = data.profile;
      state.queries = data.queries;
      state.stage = "profile";
      state.estimate = null;
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = false;
      render();
      if (state.stage === "profile") refreshEstimate();
    }
  };
  return h(
    "section",
    { class: "hero" },
    h(
      "div",
      { class: "stack" },
      h("h1", {}, "Yapay zekâ asistanları markanı öneriyor mu?"),
      h(
        "p",
        { class: "lead" },
        "Web siteni okuruz, ürünlerine göre sorular yazarız ve asistanlara sorarız. Sonra markanın listeye neden girmediğini ve açıklamana ne yazarsan seçileceğini gösteririz."
      ),
      h(
        "form",
        { class: "ask", onsubmit: submit },
        input,
        h("button", { class: "btn", type: "submit", disabled: state.busy || null }, state.busy ? "Site okunuyor…" : "Markayı tanı")
      ),
      state.error ? h("p", { class: "alert", role: "alert" }, state.error) : null,
      h("p", { class: "hint" }, "Bu adımda siteniz okunur ve marka profili çıkarılır: iki yapay zekâ çağrısı. Analiz, siz soruları onaylayınca başlar.")
    ),
    specimen()
  );
}

function specimen() {
  const answer =
    "Aidat ödemek istemiyorsanız Enpara.com kartı öne çıkıyor. Harcamalarınız yüksekse Akbank Axess ve Yapı Kredi World da iyi seçenekler.";
  return h(
    "figure",
    { class: "specimen", "aria-label": "Örnek asistan yanıtı" },
    h("span", { class: "q" }, "Aidatsız kredi kartı önerir misin?"),
    h("p", { class: "a" }, marked(answer, [], ["Enpara.com", "Akbank Axess", "Yapı Kredi World"])),
    h("p", { class: "missing" }, "Senin markan bu yanıtta yok. Kısa listeye girmek için ne gerektiğini ölçüyoruz."),
    h("figcaption", { class: "caption" }, "Örnek yanıt, gösterim amaçlı.")
  );
}

/* ---------- analyze: profile and questions ---------- */

function plan() {
  const p = state.profile;
  return {
    brand: p.brand,
    sector: p.sector,
    language: p.language || "tr",
    aliases: p.aliases || [],
    discovery: state.queries.discovery.filter((q) => q.trim()),
    named: state.queries.named.filter((q) => q.trim()),
    description: p.description || "",
    second_assistant: state.options.second,
    audit_ai: state.options.auditAi,
    reps: state.options.reps,
    max_calls: 120,
  };
}

let estimateTimer = null;
function refreshEstimate() {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(async () => {
    if (!state.profile || !plan().discovery.length) return;
    try {
      state.estimate = await api("/api/estimate", plan());
    } catch {
      state.estimate = null;
    }
    const cost = document.getElementById("cost");
    if (cost) cost.replaceWith(costLine());
  }, 250);
}

function costLine() {
  const e = state.estimate;
  const text = !e
    ? "Çağrı sayısı hesaplanıyor…"
    : e.cached
      ? `Bu analiz daha önce yapıldı; kayıtlı yanıtlar yeniden ödenmez. Tahmini ${e.calls} çağrı.`
      : `Tahmini ${e.calls} yapay zekâ ve arama çağrısı.`;
  return h("p", { class: "cost", id: "cost" }, text);
}

function questionList(kind, title, help) {
  const list = state.queries[kind];
  const wrap = h("div", { class: "qlist" });
  list.forEach((question, index) => {
    wrap.append(
      h(
        "div",
        { class: "qitem" },
        h("textarea", {
          class: "textarea",
          rows: 2,
          value: question,
          "aria-label": `${title} ${index + 1}`,
          oninput: (e) => {
            list[index] = e.target.value;
            refreshEstimate();
          },
        }),
        h("button", {
          class: "icon-btn",
          type: "button",
          "aria-label": "Soruyu kaldır",
          text: "×",
          onclick: () => {
            list.splice(index, 1);
            render();
            refreshEstimate();
          },
        })
      )
    );
  });
  const limit = kind === "discovery" ? 6 : 4;
  return h(
    "div",
    { class: "stack" },
    h("h3", {}, title),
    h("p", { class: "hint" }, help),
    wrap,
    list.length < limit
      ? h("button", {
          class: "linkish",
          type: "button",
          text: "Soru ekle",
          onclick: () => {
            list.push("");
            render();
          },
        })
      : null
  );
}

function profileView() {
  const p = state.profile;
  const text = (key, label) =>
    h(
      "label",
      { class: "field" },
      h("span", {}, label),
      h("input", {
        class: "input",
        value: key === "aliases" ? (p.aliases || []).join(", ") : p[key] || "",
        oninput: (e) => {
          p[key] = key === "aliases" ? e.target.value.split(",").map((s) => s.trim()).filter(Boolean) : e.target.value;
          refreshEstimate();
        },
      })
    );
  const start = async () => {
    state.busy = true;
    state.error = null;
    render();
    try {
      const data = await api("/api/runs", plan());
      state.job = data.id;
      state.steps = [];
      state.stage = "running";
      poll();
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = false;
      render();
    }
  };
  const rewrite = async () => {
    state.busy = true;
    render();
    try {
      state.queries = await api("/api/queries", { profile: p });
    } catch (error) {
      state.error = error.message;
    } finally {
      state.busy = false;
      render();
      refreshEstimate();
    }
  };
  const products = (p.products || []).length
    ? p.products.map((item) =>
        h(
          "div",
          { class: "product" },
          h("h3", {}, item.name),
          h("p", { class: "cat" }, item.category),
          (item.sentences || []).slice(0, 2).map((s) => h("p", { class: "quote" }, `“${s}”`))
        )
      )
    : [h("p", { class: "hint" }, "Sitede ürün anlatan bir cümle bulunamadı; açıklama incelemesi arama sonuçlarındaki özetlerle yapılır.")];

  return h(
    "section",
    {},
    steps("profile"),
    h(
      "div",
      { class: "two" },
      h(
        "div",
        { class: "panel" },
        h("h2", {}, "Markanı böyle tanıdık"),
        h(
          "p",
          { class: "hint" },
          p.website ? `Kaynak: ${p.website}` : "Web sitesi bulunamadı; profil arama sonuçlarından çıkarıldı.",
          " Yanlış olanı düzeltin."
        ),
        h("div", { class: "row3" }, text("brand", "Marka"), text("sector", "Sektör"),
          h("label", { class: "field" }, h("span", {}, "Dil"),
            h("select", { class: "select", onchange: (e) => { p.language = e.target.value; refreshEstimate(); } },
              h("option", { value: "tr", selected: p.language !== "en" || null }, "Türkçe"),
              h("option", { value: "en", selected: p.language === "en" || null }, "İngilizce")))),
        text("aliases", "Markanın diğer yazılışları"),
        h("div", { class: "stack" }, h("h3", {}, "Ürünler ve sitedeki anlatımı"), products)
      ),
      h(
        "div",
        { class: "panel" },
        h("h2", {}, "Asistanlara bunları soracağız"),
        questionList("discovery", "Markanı anmayan sorular", "Asistan markanı kendiliğinden öneriyor mu? Görünürlük bu sorularla ölçülür."),
        questionList("named", "Markanı ve ürününü anan sorular", "Adın geçtiğinde asistan seni mi, rakibini mi öneriyor?"),
        h("button", { class: "btn quiet small", type: "button", onclick: rewrite, disabled: state.busy || null }, "Soruları yeniden yaz"),
        h(
          "div",
          { class: "stack" },
          h("label", { class: "check" },
            h("input", { type: "checkbox", checked: state.options.second || null, onchange: (e) => { state.options.second = e.target.checked; refreshEstimate(); } }),
            h("span", {}, "İkinci bir asistanla karşılaştır", h("small", {}, "Aynı sorular gpt-oss-120b'ye de sorulur. Her yanıt yaklaşık 25 saniye sürer."))),
          h("label", { class: "check" },
            h("input", { type: "checkbox", checked: state.options.auditAi || null, onchange: (e) => { state.options.auditAi = e.target.checked; refreshEstimate(); } }),
            h("span", {}, "Açıklamayı yapay zekâyla incele", h("small", {}, "Cümle türlerini her sektörde doğru bulur; bir çağrı ekler.")))
        )
      )
    ),
    h(
      "div",
      { class: "go" },
      h("button", { class: "btn", type: "button", onclick: start, disabled: state.busy || !plan().discovery.length || null }, state.busy ? "Başlatılıyor…" : "Analizi başlat"),
      costLine(),
      h("button", { class: "linkish", type: "button", text: "Başka bir marka", onclick: () => { state.stage = "start"; state.error = null; render(); } })
    ),
    state.error ? h("p", { class: "alert", role: "alert", style: "margin-top:14px" }, state.error) : null
  );
}

/* ---------- analyze: running ---------- */

async function poll() {
  if (!state.job) return;
  try {
    const data = await api(`/api/runs/${state.job}`, null, "GET");
    state.steps = data.steps.map((s) => s.node);
    if (data.status === "done") {
      state.result = data.result;
      state.stage = "result";
      render();
      window.scrollTo({ top: 0 });
      return;
    }
    if (data.status === "error") {
      state.error = data.error;
      state.stage = "profile";
      render();
      return;
    }
    if (state.stage === "running" && state.view === "analyze") {
      const list = document.getElementById("run-steps");
      if (list) list.replaceWith(runSteps());
    }
  } catch (error) {
    state.error = error.message;
  }
  setTimeout(poll, 1200);
}

function runSteps() {
  const done = new Set(state.steps.map((s) => (s.startsWith("advise_") ? "report" : s)));
  const next = STEP_ORDER.find((s) => !done.has(s));
  return h(
    "ol",
    { id: "run-steps", "aria-live": "polite" },
    STEP_ORDER.map((step) =>
      h("li", { "data-state": done.has(step) && step !== next ? "done" : step === next ? "now" : "todo" }, h("i", {}), STEP_TEXT[step])
    )
  );
}

function runningView() {
  return h(
    "section",
    {},
    steps("running"),
    h(
      "div",
      { class: "run" },
      h("h2", {}, `${state.profile.brand} için analiz sürüyor`),
      h("p", { class: "hint", style: "margin-top:8px" }, state.options.second ? "İkinci asistan hız sınırıyla çalıştığı için birkaç dakika sürebilir." : "Genellikle bir iki dakika sürer."),
      runSteps()
    )
  );
}

/* ---------- analyze: result ---------- */

function figure(value, label) {
  return h("div", { class: "figure" }, h("b", {}, value), h("span", {}, label));
}

function answersSection(r) {
  const own = [r.brand, ...(state.profile?.aliases || [])];
  const assistantsSeen = [...new Set(r.answers.filter((a) => a.kind !== "named").map((a) => a.assistant))];
  let current = assistantsSeen[0] || "gemini";
  const holder = h("div", { class: "answers" });
  const fill = () => {
    holder.replaceChildren(
      ...r.answers
        .filter((a) => a.kind !== "named" && a.assistant === current)
        .map((a) =>
          h(
            "article",
            { class: "answer" },
            h("p", { class: "q" }, a.query),
            h("p", { class: "meta" }, a.mentioned ? (a.first ? "Seni ilk sırada andı" : "Seni andı, ama ilk sırada değil") : "Seni anmadı"),
            answerBody(a.answer || "", own, (a.named_brands || []).filter((b) => b !== r.brand))
          )
        )
    );
  };
  fill();
  const toggle =
    assistantsSeen.length > 1
      ? h(
          "div",
          { class: "switch", role: "group", "aria-label": "Asistan" },
          assistantsSeen.map((name) =>
            h("button", {
              type: "button",
              "aria-pressed": String(name === current),
              text: r.assistant_labels[name] || name,
              onclick: (e) => {
                current = name;
                e.target.parentElement.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.target)));
                fill();
              },
            })
          )
        )
      : null;
  return h(
    "section",
    { class: "section" },
    h("header", {}, h("h2", {}, "Asistanlar ne dedi"), h("p", {}, "Arama sonuçlarıyla verilen yanıtlar. Senin markan sarıyla, rakipler griyle işaretli.")),
    toggle,
    holder
  );
}

function recommendationsSection(r) {
  return h(
    "section",
    { class: "section" },
    h("header", {}, h("h2", {}, "Listeye girmek için"), h("p", {}, "Her öneri kontrollü testte ölçülmüş etkisiyle birlikte.")),
    h(
      "div",
      { class: "recs" },
      (r.recommendations || []).map((rec) =>
        h(
          "article",
          { class: "rec" },
          h(
            "div",
            {},
            h("h3", {}, rec.title),
            h("p", {}, rec.why),
            h("p", { class: "todo" }, rec.action),
            h("p", { class: "evidence" }, rec.verdict),
            rec.targets && rec.targets.length
              ? h(
                  "div",
                  { class: "table-wrap" },
                  h(
                    "table",
                    {},
                    h("thead", {}, h("tr", {}, h("th", {}, "Sayfa"), h("th", {}, "Andığı rakipler"), h("th", {}, "Sıra"))),
                    h(
                      "tbody",
                      {},
                      rec.targets.map((t) =>
                        h(
                          "tr",
                          {},
                          h("td", {}, h("a", { href: t.link, target: "_blank", rel: "noopener noreferrer" }, t.domain)),
                          h("td", {}, t.rivals.join(", ")),
                          h("td", {}, t.position ?? "–")
                        )
                      )
                    )
                  )
                )
              : null
          )
        )
      )
    )
  );
}

function shareBars(rows) {
  return h(
    "div",
    {},
    h("div", { class: "series" }, h("span", {}, h("i", { style: "background:var(--s1)" }), "Gemini 3.5 Flash Lite"), h("span", {}, h("i", { style: "background:var(--s2)" }), "gpt-oss-120b")),
    h(
      "div",
      { class: "bars" },
      rows.map((row) =>
        h(
          "div",
          { class: "bar-row" },
          h("span", { class: "label" }, row.label),
          h(
            "div",
            { class: "bar-track" },
            h("span", { class: "chance", style: "left:20%", title: "Rastgele seçim: %20" }),
            h("span", { class: "bar" }, h("i", { style: `width:${row.gemini * 100}%` }), pct(row.gemini)),
            h("span", { class: "bar s2" }, h("i", { style: `width:${row.cerebras * 100}%` }), pct(row.cerebras))
          )
        )
      )
    ),
    h("p", { class: "hint", style: "margin-top:10px" }, "Her kartta farklı bir cümle olduğunda o türün önerilen ürün olma payı. Kesikli çizgi rastgele seçim (%20).")
  );
}

function markedCopy(marks) {
  return h(
    "div",
    {},
    h(
      "p",
      { class: "copy" },
      marks.map((m) =>
        m.verdict
          ? [h("span", { class: "s", "data-verdict": m.verdict, title: m.labels.join(", ") }, m.text, h("span", { class: "tag" }, m.labels.join(", "))), " "]
          : [m.text, " "]
      )
    ),
    h(
      "div",
      { class: "legend" },
      h("span", {}, h("i", { style: "background:var(--good-bg);box-shadow:inset 0 -2px 0 var(--good)" }), VERDICT_TEXT.strong),
      h("span", {}, h("i", { style: "background:var(--mixed-bg);box-shadow:inset 0 -2px 0 var(--s1)" }), VERDICT_TEXT.mixed),
      h("span", {}, h("i", { style: "background:var(--weak-bg);box-shadow:inset 0 -2px 0 #9aa1b1" }), VERDICT_TEXT.weak),
      h("span", {}, h("i", { style: "background:var(--risk-bg);box-shadow:inset 0 -2px 0 var(--risk)" }), VERDICT_TEXT.risk)
    )
  );
}

function adviceList(items) {
  return h(
    "ul",
    { class: "advice" },
    items.map((item) => h("li", { "data-kind": item.kind }, h("b", {}, item.suggestion), h("span", {}, item.basis)))
  );
}

function descriptionSection(r) {
  const record = r.description_audit || {};
  if (!record.findings && !record.advice) return null;
  const source =
    record.source === "user"
      ? "Sitendeki ürün cümleleri incelendi."
      : "Sitende ürün cümlesi bulunamadığı için arama sonuçlarında markanı anlatan özetler incelendi.";
  return h(
    "section",
    { class: "section" },
    h("header", {}, h("h2", {}, "Seçilmek için: açıklaman"), h("p", {}, `Marka listedeyken asistan somut ürün bilgisine göre seçiyor. ${source}`)),
    h(
      "div",
      { class: "split" },
      h("div", {}, r.description_marks ? markedCopy(r.description_marks) : null, adviceList(record.advice || [])),
      record.findings && record.findings.length ? shareBars(record.findings) : h("p", { class: "hint" }, "Ölçülen cümle türlerinden hiçbiri bulunamadı.")
    )
  );
}

function namedSection(r) {
  const per = r.named_measures || {};
  if (!Object.keys(per).length) return null;
  const own = [r.brand, ...(state.profile?.aliases || [])];
  return h(
    "section",
    { class: "section" },
    h("header", {}, h("h2", {}, "Adın geçtiğinde"), h("p", {}, "Markanı ve ürününü anan sorular. Asistan seni mi, rakibini mi öneriyor?")),
    h(
      "div",
      { class: "figures", style: "border-top:1px solid var(--rule)" },
      Object.entries(per).map(([name, m]) =>
        figure(pct(m.picked), `${r.assistant_labels[name] || name} seni önerdi${m.rivals_picked.length ? `; yerine en çok ${m.rivals_picked[0][0]}` : ""}`)
      )
    ),
    h(
      "div",
      { class: "answers", style: "margin-top:18px" },
      r.answers
        .filter((a) => a.kind === "named" && a.assistant === "gemini")
        .map((a) =>
          h(
            "article",
            { class: "answer" },
            h("p", { class: "q" }, a.query),
            h("p", { class: "meta" }, a.picked ? (a.picked === r.brand ? "Seni önerdi" : `${a.picked} önerdi`) : "Net bir öneri yok"),
            answerBody(a.answer || "", own, (a.named_brands || []).filter((b) => b !== r.brand))
          )
        )
    )
  );
}

function rivalsSection(r) {
  const counts = {};
  r.answers
    .filter((a) => a.kind !== "named" && a.assistant === "gemini")
    .forEach((a) => (a.named_brands || []).forEach((b) => (counts[b] = (counts[b] || 0) + 1)));
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "tr")).slice(0, 8);
  if (!rows.length) return null;
  const max = rows[0][1];
  return h(
    "section",
    { class: "section" },
    h("header", {}, h("h2", {}, "Yanıtlarda kimler var"), h("p", {}, "Arama sonuçlarıyla verilen Gemini yanıtlarında markaların kaç kez anıldığı.")),
    h(
      "div",
      { class: "bars" },
      rows.map(([name, count]) =>
        h(
          "div",
          { class: "bar-row" },
          h("span", { class: "label" }, name),
          h("span", { class: `bar ${name === r.brand ? "own" : "rival"}` }, h("i", { style: `width:${(count / max) * 100}%` }), String(count))
        )
      )
    )
  );
}

function resultView() {
  const r = state.result;
  const m = r.measures || {};
  const scores = r.scores || {};
  return h(
    "section",
    {},
    steps("result"),
    h(
      "div",
      { class: "verdict" },
      h("p", { class: "who" }, `${r.brand}, ${r.sector}`),
      h("h1", {}, r.diagnosis_title),
      h("p", {}, inline(r.diagnosis_note || ""))
    ),
    h(
      "div",
      { class: "figures" },
      figure(pct(m.retrieval_presence), "arama sonuçlarında görünme"),
      figure(pct(m.mention_on), "asistan yanıtlarında anılma"),
      figure(pct(m.first_on), "ilk önerilen marka olma"),
      figure(scores.rank ? `${scores.rank}/${scores.candidates}` : "–", "rakiplere göre tahmini sıra")
    ),
    answersSection(r),
    recommendationsSection(r),
    descriptionSection(r),
    namedSection(r),
    rivalsSection(r),
    h(
      "div",
      { class: "actions" },
      h("a", { class: "btn", href: `/api/runs/${state.job}/report.pdf` }, "Raporu PDF olarak indir"),
      h("a", { class: "btn quiet", href: `/api/runs/${state.job}/report.md` }, "Markdown olarak indir"),
      h("button", { class: "btn quiet", type: "button", text: "Soruları değiştir", onclick: () => { state.stage = "profile"; render(); } }),
      h("button", { class: "btn quiet", type: "button", text: "Başka bir marka", onclick: () => { state.stage = "start"; state.result = null; render(); } })
    )
  );
}

/* ---------- audit view ---------- */

function auditView() {
  const a = state.audit;
  const submit = async (event) => {
    event.preventDefault();
    if (!a.text.trim() || a.busy) return;
    a.busy = true;
    a.error = null;
    render();
    try {
      const rivals = a.rivals.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
      a.result = await api("/api/audit", { text: a.text, rivals, ai: a.ai });
    } catch (error) {
      a.error = error.message;
    } finally {
      a.busy = false;
      render();
    }
  };
  const result = a.result;
  return h(
    "section",
    {},
    h(
      "div",
      { class: "section", style: "padding-top:10px" },
      h("header", {}, h("h1", { style: "font-size:clamp(30px,4vw,44px)" }, "Açıklaman seçilmeni sağlıyor mu?"), h("p", { class: "lead", style: "max-width:60ch" }, "Ürün bir asistanın listesindeyken seçimi açıklamadaki bilgi belirliyor. Metnini yapıştır; hangi cümlenin işe yaradığını, hangisinin yaramadığını işaretleyelim.")),
      h(
        "form",
        { class: "two", onsubmit: submit },
        h("label", { class: "field" }, h("span", {}, "Ürün açıklaman"), h("textarea", { class: "textarea", value: a.text, oninput: (e) => (a.text = e.target.value), placeholder: "60'tan fazla ülkede sunucu. WireGuard ve AES-256 şifreleme kullanır. …" })),
        h(
          "div",
          { class: "stack" },
          h("label", { class: "field" }, h("span", {}, "Rakiplerin açıklamaları (isteğe bağlı; her birini boş bir satırla ayır)"), h("textarea", { class: "textarea", value: a.rivals, oninput: (e) => (a.rivals = e.target.value) })),
          h("label", { class: "check" },
            h("input", { type: "checkbox", checked: a.ai || null, onchange: (e) => (a.ai = e.target.checked) }),
            h("span", {}, "Yapay zekâyla incele", h("small", {}, "Her sektörde doğru sonuç verir; bir çağrı yapar. Kapalıyken yalnız deneyin iki kategorisini tanıyan kurallar çalışır."))),
          h("div", {}, h("button", { class: "btn", type: "submit", disabled: a.busy || null }, a.busy ? "İnceleniyor…" : "Açıklamayı denetle"))
        )
      ),
      a.error ? h("p", { class: "alert", role: "alert", style: "margin-top:14px" }, a.error) : null
    ),
    result
      ? h(
          "div",
          { class: "section", style: "padding-top:10px" },
          result.note ? h("p", { class: "alert info" }, result.note) : null,
          result.record.findings.some((f) => f.verdict === "risk")
            ? h("p", { class: "alert", role: "alert" }, "Açıklamada kaynağı gösterilmeyen bir kurum ya da klinik iddiası var. Deneyde bu tür cümle kazandırdı ama asistanlar onu sorgulamadan kullanıcıya aktardı; kaldırın ya da kaynağını verin.")
            : null,
          h(
            "div",
            { class: "split", style: "margin-top:18px" },
            h("div", {}, markedCopy(result.marks), adviceList(result.record.advice)),
            result.record.findings.length ? shareBars(result.record.findings) : h("p", { class: "hint" }, "Ölçülen cümle türlerinden hiçbiri bulunamadı.")
          ),
          h("p", { class: "hint", style: "margin-top:18px;max-width:68ch" }, result.method_note)
        )
      : null
  );
}

/* ---------- method view ---------- */

function methodView() {
  const types = state.evidence;
  if (!types) {
    api("/api/evidence", null, "GET").then((data) => {
      state.evidence = data.types;
      if (state.view === "method") render();
    });
  }
  return h(
    "article",
    { class: "method" },
    h("h1", {}, "Asistanlar markaları neye göre öneriyor"),
    h("p", {}, "İki kontrollü deney ve 9.886 kayıtlı asistan yanıtıyla ölçtük. Sonuç iki adımda okunuyor."),
    h("h2", {}, "Önce listeye girmek"),
    h("p", {}, "Asistan aramayla bilgi topluyorsa, markanın bağımsız bir sayfada rakipleriyle birlikte ve üst sırada görünmesi belirleyici. Kontrollü testte tek bir karşılaştırma sayfası markanın anılmasını yüzde 2'den 33'e, sayfa ilk sıradayken 65'e çıkardı. Yalnız markayı öven bir sayfa işe yaramadı."),
    h("h2", {}, "Sonra seçilmek"),
    h("p", {}, "Marka listedeyken asistan ürün hakkındaki somut bilgiye bakıyor. Aynı özellikteki beş ürün kartında, ürüne özgü teknik ayrıntı ve fiyat avantajı iki farklı asistanda da en çok kazandıran cümlelerdi. \"En iyi\" demek, otorite dili ve duygusal dil kazandırmadı. Kaynağı verilmeyen kurum iddiaları kazandırdı ama asistanlar onları sorgulamadan kullanıcıya aktardı; bu araç böyle bir cümleyi hiçbir zaman önermez."),
    types ? shareBars(types) : h("p", { class: "hint" }, "Ölçümler yükleniyor…"),
    h("h2", {}, "Ölçüm nasıl yapılıyor"),
    h("p", {}, "Bir markanın anılıp anılmadığını bir yapay zekâya sormuyoruz; yanıtlarda marka adlarını eşleştirerek sayıyoruz. Rakipler arama sonuçlarından çıkarılıyor ve düzeltilebiliyor. Her ücretli çağrı kaydediliyor; aynı analiz tekrar ödenmiyor."),
    h("h2", {}, "Neyi söylemiyoruz"),
    h("p", {}, "Etkiler Gemini 3.5 Flash Lite ve gpt-oss-120b ile ölçüldü; ChatGPT ve Claude gibi asistanlarda ölçülmedi. Açıklama deneyi iki kategoriyle yapıldı; başka sektörlerde yön aynı olsa da büyüklük ölçülmedi. Kullanıcı testi yapılmadı.")
  );
}

render();
