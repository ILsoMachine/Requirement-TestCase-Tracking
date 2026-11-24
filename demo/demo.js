"use strict";

const $ = (id) => document.getElementById(id);
const projects = { ebt: "EBT", mozilla: "Mozilla 前端", edc: "Eclipse EDC" };
const methods = {
  bm25_deepseek_k10: "BM25 → DeepSeek · 前 10",
  semantic_deepseek_k10: "MiniLM → DeepSeek · 前 10",
  semantic_deepseek_k30: "MiniLM → DeepSeek · 前 30",
};
const storageKey = "requirement-test-case-demo-links-v1";
let catalog;
let result;
let selected;
let links = [];
let persistent = true;
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
  if (Array.isArray(saved)) links = saved.filter((row) =>
    row && ["project", "query", "test"].every((key) => typeof row[key] === "string"));
} catch {
  persistent = false;
}

function addOptions(select, entries, preferred) {
  select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  if (entries.some(([value]) => value === preferred)) select.value = preferred;
}

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").dataset.error = String(error);
}

function syncQuery(remembered = $("query").value) {
  const project = catalog.projects[$("project").value];
  const ids = $("mode").value === "archive" ? project.archived : Object.keys(project.queries);
  addOptions($("query"), ids.map((id) => {
    const text = project.queries[id];
    const short = text.length > 70 ? text.slice(0, 69) + "…" : text;
    return [id, `${id} · ${short}`];
  }), remembered);
  $("run").disabled = !ids.length;
  $("requirement-id").textContent = $("query").value;
  $("requirement-text").textContent = project.queries[$("query").value] || "";
}

function clearResults() {
  result = undefined;
  selected = undefined;
  $("before").replaceChildren();
  $("after").replaceChildren();
  $("before-heading").textContent = "检索前 " + catalog.display_k;
  $("after-heading").textContent = "重排前 " + catalog.display_k;
  $("before-time").textContent = "";
  $("after-time").textContent = "";
  $("after-empty").hidden = true;
  $("evidence").hidden = true;
}

function milliseconds(value) {
  if (value === null) return "";
  return value >= 1000 ? (value / 1000).toFixed(2) + " 秒" : value.toFixed(2) + " 毫秒";
}

function rankItem(id, position, isAfter) {
  const row = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.caseId = id;
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => showEvidence(id, true));
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = position;
  const description = document.createElement("span");
  description.className = "case-text";
  const code = document.createElement("strong");
  code.textContent = id;
  const text = document.createElement("span");
  text.lang = "en";
  text.textContent = result.cases[id].text;
  description.append(code, text);
  button.append(rank, description);
  if (result.after.length) {
    const { before, after } = result.cases[id];
    const delta = document.createElement("span");
    delta.className = "delta";
    const shift = before - after;
    if (shift) {
      delta.dataset.direction = shift > 0 ? "up" : "down";
      delta.textContent = (shift > 0 ? "↑" : "↓") + Math.abs(shift);
      delta.setAttribute("aria-label", `第 ${before} 名到第 ${after} 名`);
    } else {
      delta.textContent = "—";
      delta.setAttribute("aria-label", "名次持平");
    }
    if (isAfter && before > catalog.display_k) delta.textContent += " · 新进前五";
    button.append(delta);
  }
  row.append(button);
  return row;
}

function renderRankings() {
  const settings = catalog.treatments[$("treatment").value];
  const baseline = settings.baseline === "semantic" ? "MiniLM" : "BM25";
  $("before-heading").textContent = `${baseline} · 前 ${catalog.display_k}`;
  $("after-heading").textContent = `DeepSeek 重排 · 前 ${catalog.display_k}`;
  $("before-time").textContent = milliseconds(result.baseline_ms);
  $("after-time").textContent = result.total_ms === null ? "" : "总耗时 " + milliseconds(result.total_ms);
  $("before").replaceChildren(...result.before.map((id, i) => rankItem(id, i + 1, false)));
  $("after").replaceChildren(...result.after.map((id, i) => rankItem(id, i + 1, true)));
  $("after-empty").hidden = result.status !== "local";
  const source = { archive: "归档回放 · " + catalog.archive, live: "实时运行", local: "实时本地检索" };
  const state = {
    reranked: `${catalog.model} 已重排前 ${result.cutoff} 项`,
    fallback: "重排回退，保留检索顺序",
    local: "已完成候选检索",
  };
  const count = catalog.projects[$("project").value].count;
  setStatus(`${source[result.mode]}  ·  ${state[result.status]}  ·  ${count} 条候选用例。` +
    (result.notice ? " " + result.notice : ""), result.status === "fallback");
  const first = result.after.find((id) => !result.before.includes(id)) || result.after[0] || result.before[0];
  if (first) showEvidence(first);
}

function highlight(target, text, quote) {
  target.replaceChildren();
  if (!quote || !text.includes(quote)) {
    target.textContent = text;
    return;
  }
  const parts = text.split(quote);
  parts.forEach((part, index) => {
    if (index) {
      const mark = document.createElement("mark");
      mark.textContent = quote;
      target.append(mark);
    }
    target.append(part);
  });
}

function sameLink(row, project, query, test) {
  return row.project === project && row.query === query && row.test === test;
}

function showEvidence(id, scroll = false) {
  selected = id;
  document.querySelectorAll("[data-case-id]").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.caseId === id)));
  const { text, assessment } = result.cases[id];
  const grades = ["缺少关联证据", "可能相关", "直接相关"];
  $("evidence-id").textContent = id;
  $("grade").textContent = !assessment ? "暂无模型证据" :
    `有效等级 ${assessment.effective_grade} · ${grades[assessment.effective_grade]}` +
    (assessment.evidence_status === "invalid" ? " · 原文引文校验失败" : "");
  $("reason").hidden = !assessment?.reason;
  $("reason").textContent = assessment?.reason || "";
  const valid = assessment?.evidence_status === "valid";
  highlight($("requirement-quote"), result.requirement, valid && assessment.requirement_quote);
  highlight($("case-quote"), text, valid && assessment.test_quote);
  $("confirm").disabled = links.some((row) => sameLink(row, $("project").value, $("query").value, id));
  $("confirm").textContent = $("confirm").disabled ? "已确认" : "确认追踪链接";
  $("confirmation").textContent = "";
  $("evidence").hidden = false;
  if (scroll) $("evidence").scrollIntoView({ block: "nearest" });
}

function updateReverse(preferred = $("reverse").value) {
  const cases = new Map(links.map(({ project, test }) => [JSON.stringify([project, test]), `${projects[project]} · ${test}`]));
  addOptions($("reverse"), [...cases], preferred);
  if (!cases.size) addOptions($("reverse"), [["", "暂无已确认链接"]]);
  $("link-count").textContent = links.length;
  $("reverse").disabled = $("export").disabled = !cases.size;
  renderReverse();
}

function renderReverse() {
  $("reverse-results").replaceChildren();
  if (!$("reverse").value) return;
  const [project, test] = JSON.parse($("reverse").value);
  const queries = catalog.projects[project].queries;
  for (const row of links.filter((link) => link.project === project && link.test === test)) {
    const item = document.createElement("li");
    item.textContent = `${row.query} · ${queries[row.query]}`;
    $("reverse-results").append(item);
  }
}

function confirmLink() {
  const project = $("project").value;
  const query = $("query").value;
  if (!selected || !result?.cases[selected] || links.some((row) => sameLink(row, project, query, selected))) return;
  links.push({ project, query, test: selected });
  try {
    localStorage.setItem(storageKey, JSON.stringify(links));
  } catch {
    persistent = false;
  }
  updateReverse(JSON.stringify([project, selected]));
  showEvidence(selected);
  $("confirmation").textContent = persistent ? "已保存。可从下方按用例反查需求。" : "已确认。此浏览器无法持久保存，请导出 CSV。";
}

function exportCsv() {
  const quote = (value) => '"' + value.replaceAll('"', '""') + '"';
  const rows = links.map(({ project, query, test }) => [project, query, test].map(quote).join(","));
  const csv = ["dataset,requirement_id,test_case_id", ...rows].join("\r\n") + "\r\n";
  const url = URL.createObjectURL(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "confirmed-trace-links.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function run(event) {
  if (event) event.preventDefault();
  if ($("run").disabled) return;
  const fields = ["project", "query", "treatment", "mode"];
  const request = Object.fromEntries(fields.map((id) => [id, $(id).value]));
  for (const id of [...fields, "run"]) $(id).disabled = true;
  clearResults();
  setStatus(request.mode === "live" ? "正在检索并等待 DeepSeek 返回…" : "正在准备排名…");
  try {
    const response = await fetch("/api/compare", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "运行失败。");
    result = data;
    renderRankings();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    for (const id of [...fields, "run"]) $(id).disabled = false;
  }
}

async function init() {
  const response = await fetch("/api/catalog");
  if (!response.ok) throw new Error("无法加载项目。");
  catalog = await response.json();
  links = links.filter(({ project, query }) => catalog.projects[project]?.queries[query]);
  addOptions($("project"), Object.entries(projects), "ebt");
  addOptions($("treatment"), Object.entries(methods), "semantic_deepseek_k10");
  syncQuery("137");
  for (const id of ["project", "mode", "query", "treatment"]) {
    $(id).addEventListener("change", () => {
      syncQuery();
      clearResults();
      setStatus("点击「运行」查看候选。");
    });
  }
  $("controls").addEventListener("submit", run);
  $("confirm").addEventListener("click", confirmLink);
  $("reverse").addEventListener("change", renderReverse);
  $("export").addEventListener("click", exportCsv);
  updateReverse();
  await run();
}

init().catch((error) => setStatus(error.message, true));
