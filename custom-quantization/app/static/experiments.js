const applyButton = document.querySelector("#apply-quantization");
const evaluateButton = document.querySelector("#run-evaluation");
const historySelect = document.querySelector("#experiment-history");
let savedExperiment = null;
let inferenceReady = false;
let experimentBusy = false;

function selectedExperimentSettings() {
  return { bin_width: Number(input.value), tensor: tensorSelect.value, coding_method: codingMethodSelect.value };
}

function experimentSettingsLabel(settings) {
  return `${settings.tensor === "__all__" ? "Full model" : settings.tensor} · Δ ${compactNumber(settings.bin_width, 8)} · ${settings.coding_method === "huffman" ? "Huffman" : "Unary"}`;
}

function updateExperimentSelection() {
  const settings = selectedExperimentSettings();
  const valid = Number.isFinite(settings.bin_width) && settings.bin_width >= 1e-9 && settings.bin_width <= 1 && settings.tensor;
  applyButton.disabled = !inferenceReady || experimentBusy || !valid;
  document.querySelector("#experiment-selection").textContent = valid ? experimentSettingsLabel(settings) : "Choose a valid scope and bin width.";
  const changed = savedExperiment && ["bin_width", "tensor", "coding_method"].some(key => settings[key] !== savedExperiment.settings[key]);
  document.querySelector("#experiment-stale").hidden = !changed;
}

function experimentStatus(message, error = false) {
  const element = document.querySelector("#inference-status");
  element.textContent = message;
  element.classList.toggle("error-state", error);
}

function setExperimentBusy(busy) {
  experimentBusy = busy;
  evaluateButton.disabled = busy || !inferenceReady;
  historySelect.disabled = busy;
  for (const selector of ["#inference-prompt", "#evaluation-text", "#generation-tokens"]) {
    document.querySelector(selector).disabled = busy;
  }
  document.querySelector("#experiment-details").setAttribute("aria-busy", String(busy));
  updateExperimentSelection();
}

async function experimentRequest(url, body) {
  const response = await fetch(url, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join("; ") : data.detail;
    throw new Error(detail || "The model operation failed.");
  }
  return data;
}

async function refreshExperiments() {
  const data = await experimentRequest("/api/experiments");
  historySelect.replaceChildren(new Option("Choose a saved experiment…", ""));
  for (const experiment of data.experiments) {
    historySelect.add(new Option(`${experiment.id.slice(0, 8)} · ${experimentSettingsLabel(experiment.settings)}`, experiment.id));
  }
  if (savedExperiment) historySelect.value = savedExperiment.id;
}

function showExperiment(experiment) {
  savedExperiment = experiment;
  document.querySelector("#experiment-details").hidden = false;
  document.querySelector("#evaluation-results").hidden = true;
  document.querySelector("#experiment-title").textContent = `Saved experiment ${experiment.id.slice(0, 8)}`;
  document.querySelector("#experiment-settings").textContent = `${experimentSettingsLabel(experiment.settings)} · K = 0 · L off`;
  document.querySelector("#experiment-weights").textContent = `${experiment.selected_weight_count.toLocaleString()} selected weights · ${experiment.tensor_count} tensors · ${experiment.changed_weight_count.toLocaleString()} weights changed · FP32 weight file ${formatBytes(experiment.checkpoint_weight_bytes)}`;
  document.querySelector("#download-checkpoint").href = experiment.download_url;
  if (experiment.evaluation) {
    document.querySelector("#inference-prompt").value = experiment.evaluation.prompt;
    document.querySelector("#evaluation-text").value = experiment.evaluation.evaluation_text;
    document.querySelector("#generation-tokens").value = experiment.evaluation.max_new_tokens;
    showEvaluation(experiment.evaluation);
  }
  updateExperimentSelection();
}

function showEvaluation(report) {
  const original = report.original;
  const quantized = report.quantized;
  const weight = savedExperiment.weight_metrics;
  document.querySelector("#original-completion").textContent = original.completion || "(No visible completion)";
  document.querySelector("#quantized-completion").textContent = quantized.completion || "(No visible completion)";
  document.querySelector("#generation-comparison").textContent = report.comparison.identical_generation
    ? "Both models generated the same token sequence for this prompt. Loss and logit metrics can still differ."
    : "The generated token sequences differ for this prompt.";
  const rows = [
    ["Evaluation loss (nats/token)", compactNumber(original.loss, 7), compactNumber(quantized.loss, 7)],
    ["Perplexity on evaluation text", compactNumber(original.perplexity, 7), compactNumber(quantized.perplexity, 7)],
    ["Logit RMSE vs. original", "0", compactNumber(report.comparison.logit_rmse, 7)],
    ["Next-token agreement vs. original", "100%", `${report.comparison.next_token_agreement_percent.toFixed(2)}%`],
    ["Generation latency (median)", `${original.latency_ms.toFixed(2)} ms`, `${quantized.latency_ms.toFixed(2)} ms`],
    ["Generated tokens / second", original.tokens_per_second.toFixed(2), quantized.tokens_per_second.toFixed(2)],
    ["Generated tokens", String(original.generated_tokens), String(quantized.generated_tokens)],
    ["Selected weights: MAE", "0", compactNumber(weight.mae, 7)],
    ["Selected weights: RMSE", "0", compactNumber(weight.rmse, 7)],
    ["Selected weights: maximum error", "0", compactNumber(weight.max_abs_error, 7)],
    ["Selected scope: estimated storage", formatBytes(savedExperiment.storage_estimate.fp32_reference_bytes), formatBytes(savedExperiment.storage_estimate.estimated_total_bytes)],
    ["Selected scope: payload bits / weight", "32", compactNumber(savedExperiment.storage_estimate.average_code_bits_per_original_weight, 6)],
    ["Whole model: FP32 parameter bytes", formatBytes(original.model_parameter_bytes), formatBytes(quantized.model_parameter_bytes)],
  ];
  const body = document.querySelector("#inference-metrics");
  body.replaceChildren();
  for (const values of rows) {
    const row = document.createElement("tr");
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
  document.querySelector("#evaluation-method").textContent = `${report.method} Runtime speed varies between runs. FP32 parameter bytes count unique weights and exclude activations, caches, and process overhead.`;
  document.querySelector("#evaluation-input-summary").textContent = `${report.prompt_tokens} prompt tokens · ${report.evaluation_tokens} evaluation tokens · ${report.scored_tokens} scored next-token predictions · measured ${new Date(report.created_at).toLocaleString()}. Storage estimates cover only the selected scope; the downloaded snapshot stores the complete model in FP32.`;
  document.querySelector("#evaluation-results").hidden = false;
}

applyButton.addEventListener("click", async () => {
  const settings = selectedExperimentSettings();
  setExperimentBusy(true);
  experimentStatus("Applying quantization and saving a model snapshot…");
  try {
    const experiment = await experimentRequest("/api/experiments", settings);
    showExperiment(experiment);
    experimentStatus(`Experiment ${experiment.id.slice(0, 8)} saved. Run inference to compare it with the original.`);
    await refreshExperiments();
  } catch (error) {
    experimentStatus(error.message, true);
  } finally {
    setExperimentBusy(false);
  }
});

document.querySelector("#evaluation-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!savedExperiment || experimentBusy) return;
  setExperimentBusy(true);
  document.querySelector("#evaluation-results").hidden = true;
  experimentStatus("Loading both models, scoring the evaluation text, and measuring generation…");
  try {
    const report = await experimentRequest(`/api/experiments/${savedExperiment.id}/evaluate`, {
      prompt: document.querySelector("#inference-prompt").value,
      evaluation_text: document.querySelector("#evaluation-text").value,
      max_new_tokens: Number(document.querySelector("#generation-tokens").value),
    });
    savedExperiment.evaluation = report;
    showEvaluation(report);
    experimentStatus(`Comparison complete for experiment ${savedExperiment.id.slice(0, 8)}.`);
  } catch (error) {
    experimentStatus(error.message, true);
  } finally {
    setExperimentBusy(false);
  }
});

historySelect.addEventListener("change", async () => {
  if (!historySelect.value) return;
  setExperimentBusy(true);
  try {
    showExperiment(await experimentRequest(`/api/experiments/${historySelect.value}`));
    experimentStatus(`Loaded saved experiment ${savedExperiment.id.slice(0, 8)}.`);
  } catch (error) {
    experimentStatus(error.message, true);
  } finally {
    setExperimentBusy(false);
  }
});

for (const element of [input, slider, tensorSelect, codingMethodSelect]) {
  element.addEventListener("input", updateExperimentSelection);
  element.addEventListener("change", updateExperimentSelection);
}
for (const selector of ["#inference-prompt", "#evaluation-text", "#generation-tokens"]) {
  document.querySelector(selector).addEventListener("input", () => {
    document.querySelector("#evaluation-results").hidden = true;
  });
}

async function initializeExperiments() {
  updateExperimentSelection();
  try {
    const status = await experimentRequest("/api/inference/status");
    inferenceReady = status.ready;
    experimentStatus(status.detail, !status.ready);
    evaluateButton.disabled = !inferenceReady;
    updateExperimentSelection();
    await refreshExperiments();
  } catch (error) {
    experimentStatus(error.message, true);
  }
}
initializeExperiments();
