const realModelSelect = document.querySelector("#real-model-select");
const realTensorSelect = document.querySelector("#real-tensor-scope");
const realRunButton = document.querySelector("#run-real-analysis");
const realResults = document.querySelector("#real-results");
const realNote = document.querySelector("#real-method-note");
const realCacheNote = document.querySelector("#real-model-cache-status");
const realProgressWrap = document.querySelector("#real-analysis-progress-wrap");
const realProgress = document.querySelector("#real-analysis-progress");
const realStage = document.querySelector("#real-analysis-stage");
let realModelCatalog = [];
let realAnalysisJobId = null;
let realResult = null;

function realFormatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1000 && unit < units.length - 1) {
    size /= 1000;
    unit++;
  }
  return `${size.toLocaleString("en-US", { maximumFractionDigits: 2 })} ${units[unit]}`;
}

function renderRealModelList(models) {
  realModelCatalog = models;
  realModelSelect.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.name} · ${model.available_locally ? "cached" : "download on first analysis"}`;
    realModelSelect.append(option);
  }
  if (models.some((model) => model.id === "qwen3-0.6b")) realModelSelect.value = "qwen3-0.6b";
  updateRealCacheNote();
}

function updateRealCacheNote() {
  const selected = realModelCatalog.find((model) => model.id === realModelSelect.value);
  if (!selected) return;
  realCacheNote.textContent = selected.available_locally
    ? `${selected.name} is cached locally. Analysis reads checkpoint tensors directly and runs no model inference.`
    : `${selected.name} is not cached yet. The first analysis will download its public safetensors checkpoint from Hugging Face.`;
}

function resetRealTensorOptions() {
  realTensorSelect.replaceChildren();
  const option = document.createElement("option");
  option.value = "__all__";
  option.textContent = "Whole model";
  realTensorSelect.append(option);
  realTensorSelect.disabled = true;
}

function renderRealResult(data) {
  realResult = data;
  const { model, settings, metrics } = data;
  document.querySelector("#real-model-meta").textContent =
    `${model.name} · ${model.repo_id} · revision ${model.revision} · ${model.checkpoint_files} safetensors file(s) · `
    + `${model.tensor_count.toLocaleString()} tensor(s) · ${model.weights_analyzed.toLocaleString()} weights analyzed · `
    + `source dtypes ${Object.entries(model.source_dtypes).map(([dtype, count]) => `${dtype}: ${count.toLocaleString()}`).join(", ")}`;

  document.querySelector("#real-mae").textContent = metrics.mae.toExponential(4);
  document.querySelector("#real-rmse").textContent = metrics.rmse.toExponential(4);
  document.querySelector("#real-max-error").textContent = metrics.max_abs_error.toExponential(4);
  document.querySelector("#real-occupied-bins").textContent = metrics.occupied_bins.toLocaleString();
  document.querySelector("#real-bin-foot").textContent = `${metrics.original_occupied_bins.toLocaleString()} source bins before pooling`;

  const method = settings.coding_method === "huffman" ? "Huffman" : "Unary";
  document.querySelector("#real-storage-heading").textContent = `${method} storage estimate`;
  document.querySelector("#real-coding-badge").textContent = settings.coding_method === "huffman" ? "H" : "U";
  document.querySelector("#real-total-size").textContent = realFormatBytes(metrics.estimated_total_bytes);
  document.querySelector("#real-size-vs-source").textContent = `${metrics.estimated_reduction_vs_source_percent.toFixed(1)}% smaller than source weights`;
  document.querySelector("#real-payload-size").textContent = realFormatBytes(Math.ceil(metrics.coding_payload_bits / 8));
  document.querySelector("#real-table-size").textContent = realFormatBytes(Math.ceil(metrics.estimated_table_bits / 8));
  document.querySelector("#real-fp32-size").textContent = realFormatBytes(metrics.fp32_reference_bytes);
  document.querySelector("#real-source-size").textContent = realFormatBytes(metrics.source_checkpoint_bytes);
  document.querySelector("#real-fixed-size").textContent = realFormatBytes(metrics.fixed_width_payload_bytes);
  document.querySelector("#real-code-rate").textContent = `${metrics.average_code_bits_per_weight.toFixed(4)} bits`;
  document.querySelector("#real-entropy-rate").textContent = `${metrics.entropy_bits_per_weight.toFixed(4)} bits`;

  const rows = data.top_bins.map((bin, index) => `
    <tr>
      <td>${String(index + 1).padStart(2, "0")}</td>
      <td>[${(bin.index * settings.bin_width).toExponential(3)}, ${((bin.index + 1) * settings.bin_width).toExponential(3)})</td>
      <td>${bin.frequency.toLocaleString()}</td>
      <td>${bin.code_length.toLocaleString()}</td>
      <td>${bin.pooled ? "pooled" : "own bin"}</td>
    </tr>`).join("");
  document.querySelector("#real-bin-rows").innerHTML = rows;
  document.querySelector("#real-table-count").textContent = `${metrics.original_occupied_bins.toLocaleString()} BINS`;
  document.querySelector("#real-bin-footnote").textContent = `Showing the 30 most frequent input bins. Exact error covers all ${model.weights_analyzed.toLocaleString()} selected floating-point values.`;

  const previousTensor = realTensorSelect.value;
  realTensorSelect.replaceChildren();
  const whole = document.createElement("option");
  whole.value = "__all__";
  whole.textContent = "Whole model";
  realTensorSelect.append(whole);
  for (const tensorName of model.available_tensors) {
    const option = document.createElement("option");
    option.value = tensorName;
    option.textContent = tensorName;
    realTensorSelect.append(option);
  }
  realTensorSelect.disabled = false;
  if (model.selected_tensor === "__all__" && previousTensor !== "__all__" && model.available_tensors.includes(previousTensor)) {
    realTensorSelect.value = previousTensor;
  } else {
    realTensorSelect.value = model.selected_tensor;
  }

  realNote.textContent = `${data.method_note} Settings: Δ ${settings.bin_width}, K ${settings.merge_rare_bins}, L ${settings.max_code_length ?? "off"}. `
    + `Estimated size is ${metrics.estimated_reduction_vs_fp32_percent.toFixed(1)}% below the FP32 reference and ${metrics.estimated_reduction_vs_source_percent.toFixed(1)}% below source checkpoint bytes.`;
  realResults.hidden = false;
}

async function pollRealAnalysis(jobId) {
  while (realAnalysisJobId === jobId) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    const response = await fetch(`/api/real-analysis-jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not read analysis status");
    realProgress.value = job.progress || 0;
    realStage.textContent = job.stage || job.status;
    if (job.status === "complete") {
      renderRealResult(job.result);
      const cachedModel = realModelCatalog.find((model) => model.id === job.result.model.id);
      if (cachedModel) cachedModel.available_locally = true;
      const selectedOption = realModelSelect.selectedOptions[0];
      if (selectedOption) selectedOption.textContent = `${job.result.model.name} · cached`;
      updateRealCacheNote();
      document.querySelector("#status").textContent = `${job.result.model.name} · ${job.result.model.weights_analyzed.toLocaleString()} real weights analyzed`;
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "Real-model analysis failed");
  }
}

async function runRealAnalysis() {
  if (!realModelSelect.value) return;
  const binWidth = Number(document.querySelector("#real-bin-width").value);
  const mergeRareBins = Number(document.querySelector("#real-merge-rare-bins").value);
  const maxCodeLengthRaw = document.querySelector("#real-max-code-length").value.trim();
  const maxCodeLength = maxCodeLengthRaw === "" ? null : Number(maxCodeLengthRaw);
  if (!Number.isFinite(binWidth) || binWidth < 1e-5 || binWidth > 1) {
    realNote.textContent = "Choose a finite bin width from 1e-5 to 1.";
    return;
  }
  if (!Number.isInteger(mergeRareBins) || mergeRareBins < 0 || mergeRareBins > 100000) {
    realNote.textContent = "K must be a whole number from 0 to 100,000.";
    return;
  }
  if (maxCodeLength !== null && (!Number.isInteger(maxCodeLength) || maxCodeLength < 1 || maxCodeLength > 64)) {
    realNote.textContent = "L must be 1–64, or blank to leave it off.";
    return;
  }

  const params = new URLSearchParams({
    bin_width: String(binWidth),
    merge_rare_bins: String(mergeRareBins),
    coding_method: document.querySelector("#real-coding-method").value,
    tensor: realTensorSelect.value || "__all__",
  });
  if (maxCodeLength !== null) params.set("max_code_length", String(maxCodeLength));
  realResults.hidden = true;
  realProgressWrap.hidden = false;
  realProgress.value = 0;
  realStage.textContent = "Queueing real checkpoint analysis…";
  realNote.textContent = "Opening the real checkpoint. No model forward pass or inference will run.";
  realRunButton.disabled = true;
  realRunButton.textContent = "Analyzing checkpoint…";
  try {
    const response = await fetch(`/api/real-models/${realModelSelect.value}/analysis?${params}`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not start real-model analysis");
    realAnalysisJobId = data.job_id;
    await pollRealAnalysis(data.job_id);
  } catch (error) {
    realNote.textContent = error.message;
    realStage.textContent = "Analysis failed";
  } finally {
    realAnalysisJobId = null;
    realRunButton.disabled = false;
    realRunButton.textContent = "Analyze real checkpoint";
  }
}

async function loadRealModels() {
  try {
    const response = await fetch("/api/real-models");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not load model registry");
    renderRealModelList(data.models);
  } catch (error) {
    realCacheNote.textContent = error.message;
  }
}

realRunButton.addEventListener("click", runRealAnalysis);
realModelSelect.addEventListener("change", () => {
  updateRealCacheNote();
  resetRealTensorOptions();
});
document.querySelectorAll(".view-switch-button").forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.viewTarget;
    document.querySelectorAll(".view-switch-button").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    document.querySelector("#weight-view").hidden = target !== "weight-view";
    document.querySelector("#real-model-view").hidden = target !== "real-model-view";
    document.querySelector("#dataset-title").textContent = target === "real-model-view" ? "Real Qwen checkpoints" : "Tiny GPT-2";
    document.querySelector("#dataset-subtitle").textContent = target === "real-model-view"
      ? "0.6B · 1.7B · 4B actual weights"
      : modelInfo
        ? `${modelInfo.tensor_count} tensors · ${modelInfo.total_values.toLocaleString()} weights`
        : "Loading model weights…";
  });
});

loadRealModels();
