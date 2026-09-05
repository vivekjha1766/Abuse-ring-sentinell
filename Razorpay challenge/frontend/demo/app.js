// Fraud monitoring demo dashboard. Talks only to this project's own
// backend API (same-origin, served by FastAPI's StaticFiles mount at
// /demo), which itself never reads data/splits/heldout. No build step:
// plain fetch() against relative URLs.

const API = ""; // same-origin

function $(selector) { return document.querySelector(selector); }
function $all(selector) { return Array.from(document.querySelectorAll(selector)); }

$all(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $all(".tab-btn").forEach((b) => b.classList.remove("active"));
    $all(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#" + btn.dataset.tab).classList.add("active");
  });
});

async function getJSON(path, options) {
  const response = await fetch(API + path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

function riskBadge(band) {
  return `<span class="risk-badge risk-${band}">${band}</span>`;
}

function metricsTableHTML(models, highlightType) {
  const rows = models
    .map((m) => {
      const cls = m.model_type === highlightType ? "highlight" : "";
      return `<tr class="${cls}">
        <td>${m.model_type}</td>
        <td>${m.pr_auc.toFixed(4)}</td>
        <td>${m.roc_auc.toFixed(4)}</td>
        <td>${m.precision.toFixed(4)}</td>
        <td>${m.recall.toFixed(4)}</td>
        <td>${m.f1.toFixed(4)}</td>
        <td>$${m.total_error_cost.toLocaleString()}</td>
      </tr>`;
    })
    .join("");
  return `<thead><tr>
      <th>Model</th><th>PR-AUC (primary)</th><th>ROC-AUC</th>
      <th>Precision</th><th>Recall</th><th>F1</th><th>Benchmark cost</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

// ---- Executive Dashboard ---------------------------------------------

async function loadDashboard() {
  try {
    const stats = await getJSON("/stats");
    if (stats.dataset.is_synthetic_fallback) {
      $("#dashboard-banner").innerHTML =
        "No generated dataset found locally (data/ is git-ignored; run <code>make data</code>). " +
        "Showing a small illustrative fallback fixture instead.";
    } else {
      $("#dashboard-banner").innerHTML = "";
    }
    $("#dashboard-cards").innerHTML = `
      <div class="card"><div class="value">${stats.dataset.transactions_loaded}</div><div class="label">Transactions loaded</div></div>
      <div class="card"><div class="value">${stats.dataset.accounts_loaded}</div><div class="label">Accounts loaded</div></div>
      <div class="card"><div class="value">92.7%</div><div class="label">XGBoost heldout recall</div></div>
      <div class="card"><div class="value">$107,535</div><div class="label">XGBoost heldout benchmark cost</div></div>
    `;
    $("#dashboard-metrics").innerHTML = metricsTableHTML(stats.heldout_evaluation.models, "xgboost");
    window.__heldout = stats.heldout_evaluation;
    populatePerformanceTab(stats.heldout_evaluation.models);
  } catch (err) {
    $("#dashboard-banner").innerHTML = `API not reachable: ${err.message}. Is the backend running?`;
  }
}

// ---- Model Performance --------------------------------------------------

let performanceChart = null;

function populatePerformanceTab(models) {
  $("#performance-table").innerHTML = metricsTableHTML(models, "xgboost");
  const ctx = $("#performance-chart").getContext("2d");
  const labels = models.map((m) => m.model_type);
  const data = {
    labels,
    datasets: [
      { label: "PR-AUC (primary)", data: models.map((m) => m.pr_auc), backgroundColor: "#4c6ef5" },
      { label: "ROC-AUC", data: models.map((m) => m.roc_auc), backgroundColor: "#adb5bd" },
      { label: "F1", data: models.map((m) => m.f1), backgroundColor: "#82c91e" },
    ],
  };
  if (performanceChart) performanceChart.destroy();
  performanceChart = new Chart(ctx, {
    type: "bar",
    data,
    options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } },
  });
}

// ---- Transaction Risk Explorer ------------------------------------------

async function scoreTransaction(transactionId) {
  const container = $("#explorer-result");
  container.innerHTML = "Scoring...";
  try {
    const [prediction, transaction] = await Promise.all([
      getJSON("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transaction_id: transactionId }),
      }),
      getJSON(`/transaction/${encodeURIComponent(transactionId)}`),
    ]);
    container.innerHTML = `
      <div class="cards">
        <div class="card"><div class="value">${(prediction.fraud_probability * 100).toFixed(2)}%</div><div class="label">Fraud probability</div></div>
        <div class="card"><div class="value">${prediction.risk_score} ${riskBadge(prediction.risk_band)}</div><div class="label">Risk score (0-100)</div></div>
        <div class="card"><div class="value">${prediction.decision.toUpperCase()}</div><div class="label">Decision (threshold ${prediction.threshold.toFixed(4)})</div></div>
      </div>
      <h3>Top risk signals</h3>
      <ul class="signal-list">${prediction.top_risk_signals.map((s) => `<li>${s}</li>`).join("")}</ul>
      <h3>Transaction details</h3>
      <table class="metrics-table"><tbody>
        <tr><td>Account</td><td>${transaction.account_id}</td></tr>
        <tr><td>Customer</td><td>${transaction.customer_id}</td></tr>
        <tr><td>Amount</td><td>${transaction.amount}</td></tr>
        <tr><td>Channel</td><td>${transaction.channel}</td></tr>
        <tr><td>Timestamp</td><td>${transaction.timestamp}</td></tr>
      </tbody></table>
    `;
    return prediction;
  } catch (err) {
    container.innerHTML = `Error: ${err.message}`;
    throw err;
  }
}

$("#txn-score-btn").addEventListener("click", () => {
  const id = $("#txn-input").value.trim();
  if (id) scoreTransaction(id);
});
$("#txn-sample-select").addEventListener("change", (e) => {
  if (e.target.value) {
    $("#txn-input").value = e.target.value;
    scoreTransaction(e.target.value);
  }
});

// ---- Fraud Ring Explorer ------------------------------------------------

let cy = null;

async function loadGraph(transactionId) {
  const data = await getJSON(`/graph/${encodeURIComponent(transactionId)}?radius=2`);
  const elements = [
    ...data.nodes.map((n) => ({ data: { id: n.id, label: `${n.type}:${n.label}` } })),
    ...data.edges.map((e) => ({ data: { source: e.source, target: e.target, label: e.relation } })),
  ];
  if (cy) cy.destroy();
  cy = cytoscape({
    container: $("#cy"),
    elements,
    style: [
      { selector: "node", style: { label: "data(label)", "font-size": 8, "background-color": "#4c6ef5" } },
      { selector: "edge", style: { label: "data(label)", "font-size": 7, width: 1, "line-color": "#adb5bd" } },
    ],
    layout: { name: "cose" },
  });
}

async function loadRings() {
  const data = await getJSON("/rings");
  const rows = data.rings
    .map(
      (r) => `<tr>
        <td>${r.component_id}</td><td>${r.account_count}</td><td>${r.device_count}</td>
        <td>${r.ip_count}</td><td>${r.transaction_count}</td>
        <td>${(r.fraud_concentration * 100).toFixed(1)}%</td>
      </tr>`
    )
    .join("");
  $("#rings-table").innerHTML = `<thead><tr>
      <th>Component</th><th>Accounts</th><th>Devices</th><th>IPs</th><th>Transactions</th><th>Fraud concentration</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

$("#graph-load-btn").addEventListener("click", () => {
  const id = $("#graph-txn-input").value.trim();
  if (id) loadGraph(id);
});

// ---- Demo Investigation walkthrough -------------------------------------

async function runDemoWalkthrough() {
  const steps = $("#demo-steps");
  steps.innerHTML = "";
  function addStep(html) {
    const li = document.createElement("li");
    li.innerHTML = html;
    steps.appendChild(li);
  }

  try {
    const stats = await getJSON("/stats");
    // Prefer a transaction we already know exists.
    const sampleId = window.__sampleIds && window.__sampleIds[0];
    if (!sampleId) {
      addStep("No sample transaction available -- load the Executive Dashboard first.");
      return;
    }
    addStep(`<b>1. Transaction selected:</b> ${sampleId}`);

    const prediction = await getJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: sampleId }),
    });
    addStep(
      `<b>2. Model risk score:</b> ${prediction.risk_score}/100 (${prediction.risk_band}), ` +
        `fraud probability ${(prediction.fraud_probability * 100).toFixed(2)}%, decision: ${prediction.decision}`
    );
    addStep(`<b>3. Explanation:</b> ${prediction.top_risk_signals.join("; ")}`);

    const graphData = await getJSON(`/graph/${encodeURIComponent(sampleId)}?radius=1`);
    addStep(
      `<b>4. Connected entities:</b> ${graphData.nodes.length} nodes, ${graphData.edges.length} edges ` +
        "around this transaction's account."
    );

    const ringsData = await getJSON("/rings");
    const topRing = ringsData.rings[0];
    addStep(
      topRing
        ? `<b>5. Suspicious ring candidate:</b> ${topRing.account_count} accounts, ${topRing.device_count} devices, ${topRing.ip_count} IPs, fraud concentration ${(topRing.fraud_concentration * 100).toFixed(1)}%`
        : "<b>5. Suspicious ring candidate:</b> none found in this dataset sample."
    );

    addStep(
      "<b>6. Investigator conclusion:</b> use the risk score, explanation, and ring evidence together " +
        "to decide whether to escalate for human review -- this system never blocks a customer autonomously."
    );
  } catch (err) {
    addStep(`Error running walkthrough: ${err.message}`);
  }
}

$("#demo-run-btn").addEventListener("click", runDemoWalkthrough);

// ---- Bootstrap ------------------------------------------------------------

async function bootstrap() {
  await loadDashboard();
  try {
    const stats = await getJSON("/stats");
    // Best-effort sample ids for convenience dropdowns; falls back silently.
    const alerts = await getJSON("/alerts?limit=10");
    const ids = alerts.alerts.map((a) => a.transaction_id);
    window.__sampleIds = ids.length ? ids : ["demo_txn_002"];
    const select = $("#txn-sample-select");
    window.__sampleIds.forEach((id) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      select.appendChild(opt);
    });
  } catch (err) {
    window.__sampleIds = ["demo_txn_002"];
  }
}

bootstrap();
