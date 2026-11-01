const input = document.querySelector("#bin-width");
const slider = document.querySelector("#bin-slider");
const tensorSelect = document.querySelector("#tensor-select");
const rareBinCountInput = document.querySelector("#rare-bin-count");
const codingMethodSelect = document.querySelector("#coding-method");
const maxCodeLengthInput = document.querySelector("#max-code-length");
const frequencyOrderSelect = document.querySelector("#frequency-order");
const frequencyScroll = document.querySelector("#frequency-scroll");
const tooltip = document.querySelector("#chart-tooltip");
const canvas = document.querySelector("#histogram");
const context = canvas.getContext("2d");
const huffmanReviewButton = document.querySelector("#review-huffman-codes");
const huffmanDialog = document.querySelector("#huffman-dialog");
const huffmanCodeScroll = document.querySelector("#huffman-code-scroll");
const huffmanCodeSearch = document.querySelector("#huffman-code-search");
const huffmanCodeRows = document.querySelector("#huffman-code-rows");
const binWidthSweepButton = document.querySelector("#open-bin-width-sweep");
const binWidthSweepDialog = document.querySelector("#bin-width-sweep-dialog");
const binWidthSweepRange = document.querySelector("#sweep-point-index");
const bpeButton = document.querySelector("#open-bpe-view");
const bpeDialog = document.querySelector("#bpe-dialog");
const bpeMergeInput = document.querySelector("#bpe-merge-limit");
const bpeRunButton = document.querySelector("#bpe-run-analysis");

const MIN_WIDTH = 1e-7;
const MAX_WIDTH = 1e-2;
const REFERENCE_WIDTH = 2.79e-6;
const LOG_RANGE = Math.log(MAX_WIDTH / MIN_WIDTH);
const FREQUENCY_ROW_HEIGHT = 40;
const HUFFMAN_ROW_HEIGHT = 48;
const TABLE_OVERSCAN = 5;
const SWEEP_SVG_NS = "http://www.w3.org/2000/svg";
let activeData = null;
let modelInfo = null;
let requestTimer = null;
let renderTimer = null;
let tableRenderQueued = false;
let huffmanData = null;
let huffmanVisibleRows = [];
let huffmanRenderQueued = false;
let huffmanReviewLoading = false;
let analysisRequestId = 0;
let binWidthSweepData = null;
let binWidthSweepLoading = false;
let bpeData = null;
let bpeTableKind = "bins";
let bpePage = 0;
const BPE_PAGE_SIZE = 100;

function compactNumber(value, digits = 4) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 1e6) return value.toExponential(digits - 1);
  return value.toLocaleString("en-US", { maximumSignificantDigits: digits });
}

function formatBytes(bytes) {
  if (bytes < 1000) return `${bytes.toLocaleString()} B`;
  if (bytes < 1e6) return `${(bytes / 1000).toFixed(1)} kB`;
  return `${(bytes / 1e6).toFixed(2)} MB`;
}

function formatBits(bits) {
  if (bits < 1000) return `${bits.toLocaleString()} bit`;
  if (bits < 1e6) return `${(bits / 1000).toFixed(1)} kbit`;
  return `${(bits / 1e6).toFixed(2)} Mbit`;
}

function formatPayloadSize(bits) {
  return `${formatBytes(Math.ceil(bits / 8))} · ${formatBits(bits)}`;
}

function formatRelativeSize(percentSmaller) {
  if (Math.abs(percentSmaller) < 0.05) return "Same size as FP32";
  return percentSmaller > 0
    ? `${percentSmaller.toFixed(1)}% smaller`
    : `${Math.abs(percentSmaller).toFixed(1)}% larger`;
}

function widthFromSlider(value) {
  return MIN_WIDTH * Math.exp(LOG_RANGE * Number(value) / 1000);
}

function sliderFromWidth(width) {
  const clamped = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));
  return Math.round(1000 * Math.log(clamped / MIN_WIDTH) / LOG_RANGE);
}

function setWidth(width, updateSlider = true) {
  if (!Number.isFinite(width) || width < 1e-9 || width > 1) return;
  input.value = Number(width.toPrecision(8));
  if (updateSlider) slider.value = sliderFromWidth(width);
  document.querySelector("#width-multiple").textContent = `${compactNumber(width / REFERENCE_WIDTH, 4)}× reference`;
  queueAnalysis();
}

function queueAnalysis() {
  clearTimeout(requestTimer);
  requestTimer = setTimeout(loadAnalysis, 160);
}

async function loadAnalysis() {
  const requestId = ++analysisRequestId;
  if (!modelInfo) return null;
  const width = Number(input.value);
  if (!Number.isFinite(width) || width <= 0 || width > 1) {
    setStatus("Enter a width from 1e-9 to 1", true);
    return null;
  }
  const mergeRareBins = Number(rareBinCountInput.value);
  if (!Number.isInteger(mergeRareBins) || mergeRareBins < 0) {
    setStatus("Enter a non-negative whole number of rare bins", true);
    return null;
  }
  const maxCodeLengthRaw = maxCodeLengthInput.value.trim();
  const maxCodeLength = maxCodeLengthRaw === "" ? null : Number(maxCodeLengthRaw);
  if (maxCodeLength !== null && (!Number.isInteger(maxCodeLength) || maxCodeLength < 1 || maxCodeLength > 64)) {
    setStatus("Enter a maximum code length from 1 to 64, or leave it blank", true);
    return null;
  }
  setStatus("Recalculating…");
  try {
    const params = new URLSearchParams({
      bin_width: String(width),
      tensor: tensorSelect.value,
      merge_rare_bins: String(mergeRareBins),
      frequency_order: frequencyOrderSelect.value,
      coding_method: codingMethodSelect.value,
    });
    if (maxCodeLength !== null) params.set("max_code_length", String(maxCodeLength));
    const response = await fetch(`/api/analysis?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed");
    if (requestId !== analysisRequestId) return null;
    activeData = data;
    render(data);
    const scopeLabel = data.dataset.scope === "__all__" ? "Full model" : data.dataset.scope;
    setStatus(`${scopeLabel} · ${data.dataset.values.toLocaleString()} values analyzed`);
    return data;
  } catch (error) {
    if (requestId === analysisRequestId) setStatus(error.message, true);
    return null;
  }
}

function setStatus(message, isError = false) {
  const status = document.querySelector("#status");
  status.textContent = message;
  status.classList.toggle("error-state", isError);
}

function render(data) {
  const m = data.metrics;
  const isHuffman = data.coding_method === "huffman";
  const methodName = isHuffman ? "Huffman" : "Unary";
  const methodLower = methodName.toLowerCase();
  const fullModel = data.dataset.scope === "__all__";
  const shapeLabel = data.dataset.shape ? data.dataset.shape.join(" × ") : "combined tensors";
  rareBinCountInput.max = String(m.original_occupied_bins);
  if (Number(rareBinCountInput.value) !== m.merged_rare_bin_count) {
    rareBinCountInput.value = String(m.merged_rare_bin_count);
  }
  const mergeSummary = document.querySelector("#merge-summary");
  const mergeDetails = document.querySelector("#merge-details");
  if (m.merged_rare_bin_count > 0) {
    mergeSummary.classList.add("active");
    const rareDestination = m.rare_bucket_absorbed_by_length_limit
      ? "included in the code-length overflow bucket"
      : "pooled into one shared bucket";
    mergeSummary.textContent = `${m.merged_rare_bin_count.toLocaleString()} of ${m.original_occupied_bins.toLocaleString()} least-frequent bins ${rareDestination} · ${m.merged_value_count.toLocaleString()} weights`;
    mergeDetails.hidden = false;
    document.querySelector("#merged-bin-preview").innerHTML = data.merged_bin_preview.map((bin) =>
      `<li>bin ${bin.index.toLocaleString()} · ${bin.count.toLocaleString()} weights · [${compactNumber(bin.lower, 5)}, ${compactNumber(bin.upper, 5)})</li>`
    ).join("");
    document.querySelector("#merged-preview-note").textContent = m.merged_rare_bin_count > data.merged_bin_preview.length
      ? `Previewing the ${data.merged_bin_preview.length} least frequent of ${m.merged_rare_bin_count.toLocaleString()} pooled bins.`
      : `All ${m.merged_rare_bin_count} pooled bins are shown, ordered from least frequent.`;
  } else {
    mergeSummary.classList.remove("active");
    mergeSummary.textContent = `No rare bins pooled · ${m.original_occupied_bins.toLocaleString()} occupied bins available.`;
    mergeDetails.hidden = true;
    document.querySelector("#merged-bin-preview").replaceChildren();
    document.querySelector("#merged-preview-note").textContent = "";
  }
  const lengthSummary = document.querySelector("#length-limit-summary");
  const lengthDetails = document.querySelector("#length-limit-details");
  if (m.max_code_length === null) {
    lengthSummary.classList.remove("active");
    lengthSummary.textContent = `No code-length limit · longest ${methodLower} code is ${m.actual_max_code_length} bits.`;
    lengthDetails.hidden = true;
    document.querySelector("#length-limited-bin-preview").replaceChildren();
    document.querySelector("#length-limited-preview-note").textContent = "";
  } else if (m.length_limited_bin_count > 0) {
    lengthSummary.classList.add("active");
    lengthSummary.textContent = `${m.length_limited_bin_count.toLocaleString()} source bins · ${m.length_limited_value_count.toLocaleString()} weights pooled · longest code now ${m.actual_max_code_length} bits (limit ${m.max_code_length}).`;
    lengthDetails.hidden = false;
    document.querySelector("#length-limited-bin-preview").innerHTML = data.length_limited_bin_preview.map((bin) =>
      `<li>bin ${bin.index.toLocaleString()} · ${bin.count.toLocaleString()} weights · [${compactNumber(bin.lower, 5)}, ${compactNumber(bin.upper, 5)})</li>`
    ).join("");
    document.querySelector("#length-limited-preview-note").textContent = m.length_limited_bin_count > data.length_limited_bin_preview.length
      ? `Previewing ${data.length_limited_bin_preview.length} of ${m.length_limited_bin_count.toLocaleString()} pooled bins.`
      : `All ${m.length_limited_bin_count} pooled bins are shown.`;
  } else {
    lengthSummary.classList.remove("active");
    lengthSummary.textContent = `All codes fit within ${m.max_code_length} bits · longest code is ${m.actual_max_code_length} bits; no bins pooled.`;
    lengthDetails.hidden = true;
    document.querySelector("#length-limited-bin-preview").replaceChildren();
    document.querySelector("#length-limited-preview-note").textContent = "";
  }
  document.querySelector("#dataset-title").textContent = "Tiny GPT-2";
  document.querySelector("#dataset-subtitle").textContent = fullModel
    ? `${data.dataset.tensor_count} tensors · ${data.dataset.values.toLocaleString()} weights`
    : `${shapeLabel} · ${data.dataset.values.toLocaleString()} weights`;
  document.querySelector("#scope-summary").textContent = fullModel
    ? `All ${data.dataset.tensor_count} unique learned tensors · masks and tied aliases counted once.`
    : `${shapeLabel} · ${data.dataset.tensor_count === 1 ? "one tensor" : `${data.dataset.tensor_count} tensors`} selected.`;
  document.querySelector("#source-note").textContent = fullModel
    ? `Analyzing ${data.dataset.values.toLocaleString()} unique learned weights across ${data.dataset.tensor_count} tensors. Attention masks and tied aliases are excluded.`
    : `Analyzing ${data.dataset.values.toLocaleString()} weights from ${data.dataset.scope}.`;
  document.querySelector("#coding-method-kicker").textContent = `${methodName.toUpperCase()} CODING`;
  document.querySelector("#coding-method-summary").textContent = isHuffman
    ? "Frequency-weighted, optimal binary prefix codes"
    : "Frequency-ranked unary zero-run codes";
  document.querySelector("#coding-method-detail").textContent = isHuffman
    ? "Each output bucket is a symbol. Pooled source bins share a code."
    : "The most frequent bucket gets 1; each next rank adds one zero before the 1.";
  document.querySelector("#coding-method-note").textContent = isHuffman
    ? "Huffman builds optimal binary prefix codes from bucket frequencies."
    : "Unary gives shorter zero-run codes to more frequent buckets.";
  syncHuffmanReviewButton();
  document.querySelector("#storage-heading").textContent = `${methodName} storage`;
  document.querySelector("#coding-badge").textContent = isHuffman ? "H" : "U";
  document.querySelector("#payload-label").textContent = `${methodName} payload`;
  document.querySelector("#table-label").textContent = isHuffman ? "Tree + bin table*" : "Code map + bin table*";
  document.querySelector("#storage-estimate-label").textContent = `${methodName} estimate`;
  document.querySelector("#estimate-note").textContent = `* Total size includes payload and this table estimate: ${isHuffman ? "tree markers, " : ""}output bin IDs, Δ, the minimum bin index, and pooled-bucket metadata. Tensor-shape metadata, container framing, and byte padding are excluded.`;
  document.querySelector("#mae").textContent = compactNumber(m.mae, 5);
  document.querySelector("#rmse").textContent = compactNumber(m.rmse, 5);
  document.querySelector("#occupied-bins").textContent = m.occupied_bins.toLocaleString();
  const poolingNotes = [];
  if (m.merged_rare_bin_count > 0) poolingNotes.push(`${m.merged_rare_bin_count.toLocaleString()} by K`);
  if (m.additional_length_limited_bin_count > 0) poolingNotes.push(`${m.additional_length_limited_bin_count.toLocaleString()} by code cap`);
  document.querySelector("#bin-foot").textContent = poolingNotes.length
    ? `${m.original_occupied_bins.toLocaleString()} input bins · ${poolingNotes.join(" · ")}`
    : `${m.original_occupied_bins.toLocaleString()} input symbols`;
  document.querySelector("#total-size").textContent = formatBytes(m.estimated_total_bytes);
  document.querySelector("#size-vs-fp32").textContent = `${m.estimated_fp32_reduction_percent.toFixed(1)}% smaller than FP32`;
  document.querySelector("#payload-size").textContent = `${formatBytes(Math.ceil(m.coding_payload_bits / 8))} · ${formatBits(m.coding_payload_bits)}`;
  document.querySelector("#table-size").textContent = `${formatBytes(Math.ceil(m.estimated_table_bits / 8))} · ${formatBits(m.estimated_table_bits)}`;
  document.querySelector("#fp32-size").textContent = formatBytes(m.fp32_reference_bytes);
  document.querySelector("#huffman-rate").textContent = `${m.average_code_bits_per_original_weight.toFixed(3)} bits/weight`;
  document.querySelector("#entropy-rate").textContent = `${m.entropy_bits_per_weight.toFixed(3)} bits/weight`;
  const totalRatio = Math.min(1, (m.estimated_total_bits / m.fp32_reference_bits) * 100);
  document.querySelector("#huffman-bar").style.width = `${Math.max(.5, totalRatio)}%`;
  document.querySelector("#axis-min").textContent = compactNumber(data.histogram.min_position, 4);
  document.querySelector("#axis-max").textContent = compactNumber(data.histogram.max_position, 4);
  renderComparison(data);
  renderTable(data);
  drawHistogram();
  syncBinWidthSweepButton();
}

function syncHuffmanReviewButton() {
  const huffmanSelected = codingMethodSelect.value === "huffman";
  huffmanReviewButton.hidden = !huffmanSelected;
  huffmanReviewButton.disabled = !modelInfo || huffmanReviewLoading;
}

function syncBinWidthSweepButton() {
  binWidthSweepButton.disabled = !modelInfo || !activeData || binWidthSweepLoading;
}

function huffmanSettingsMatch(data) {
  if (!data || data.coding_method !== "huffman") return false;
  const selectedWidth = Number(input.value);
  const widthMatches = Math.abs(data.bin_width - selectedWidth) <= Math.max(1e-15, Math.abs(selectedWidth) * 1e-12);
  const selectedK = Number(rareBinCountInput.value);
  const effectiveK = Math.min(selectedK, data.metrics.original_occupied_bins);
  const selectedLengthText = maxCodeLengthInput.value.trim();
  const selectedLength = selectedLengthText === "" ? null : Number(selectedLengthText);
  return widthMatches
    && data.dataset.scope === tensorSelect.value
    && data.metrics.merged_rare_bin_count === effectiveK
    && data.metrics.max_code_length === selectedLength
    && data.frequency_order === frequencyOrderSelect.value
    && codingMethodSelect.value === "huffman";
}

function openHuffmanPreview(data) {
  huffmanData = data;
  huffmanCodeSearch.value = "";
  const metrics = data.metrics;
  const scope = data.dataset.scope === "__all__" ? "Full model" : data.dataset.scope;
  const maxLength = metrics.max_code_length === null ? "off" : `${metrics.max_code_length} bits`;
  document.querySelector("#huffman-preview-config").textContent =
    `${scope} · Δ ${compactNumber(data.bin_width, 7)} · K ${metrics.merged_rare_bin_count.toLocaleString()} · L ${maxLength} · ${data.dataset.values.toLocaleString()} weights · ${metrics.original_occupied_bins.toLocaleString()} input bins`;
  document.querySelector("#huffman-preview-symbols").textContent = metrics.occupied_bins.toLocaleString();
  document.querySelector("#huffman-preview-payload").textContent = formatPayloadSize(metrics.coding_payload_bits);
  document.querySelector("#huffman-preview-rate").textContent = `${metrics.average_code_bits_per_original_weight.toFixed(3)} bits / weight`;
  document.querySelector("#huffman-preview-table-size").textContent = formatPayloadSize(metrics.estimated_table_bits);
  document.querySelector("#huffman-preview-longest").textContent = `${metrics.actual_max_code_length} bits`;

  const total = data.frequency_bins.indices.length;
  huffmanVisibleRows = Array.from({ length: total }, (_, index) => index);
  document.querySelector("#huffman-code-foot").textContent = `${total.toLocaleString()} source bins · ${data.frequency_order === "desc" ? "most-to-least" : "least-to-most"} frequent`;
  if (!huffmanDialog.open) huffmanDialog.showModal();
  huffmanCodeScroll.scrollTop = 0;
  renderHuffmanPreviewRows();
  huffmanCodeSearch.focus({ preventScroll: true });
}

function filterHuffmanPreview() {
  if (!huffmanData) return;
  const bins = huffmanData.frequency_bins;
  const query = huffmanCodeSearch.value.trim().toLowerCase();
  huffmanVisibleRows = [];
  for (let row = 0; row < bins.indices.length; row++) {
    if (!query) {
      huffmanVisibleRows.push(row);
      continue;
    }
    const index = bins.indices[row];
    const searchable = `${index} ${bins.codes[row]} ${bins.pool_reasons[row]}`.toLowerCase();
    if (searchable.includes(query)) huffmanVisibleRows.push(row);
  }
  huffmanCodeScroll.scrollTop = 0;
  renderHuffmanPreviewRows();
}

function appendHuffmanSpacer(height) {
  if (height <= 0) return;
  const row = document.createElement("tr");
  row.className = "huffman-preview-spacer";
  row.setAttribute("aria-hidden", "true");
  const cell = document.createElement("td");
  cell.colSpan = 6;
  cell.style.height = `${height}px`;
  row.append(cell);
  huffmanCodeRows.append(row);
}

function appendHuffmanCodeRow(rowIndex, rank) {
  const bins = huffmanData.frequency_bins;
  const index = bins.indices[rowIndex];
  const count = bins.counts[rowIndex];
  const codeLength = bins.code_lengths[rowIndex];
  const code = bins.codes[rowIndex];
  const pooled = bins.pooled[rowIndex];
  const lower = index * huffmanData.bin_width;
  const upper = (index + 1) * huffmanData.bin_width;
  const row = document.createElement("tr");
  row.className = "huffman-preview-row";

  const binCell = document.createElement("td");
  binCell.textContent = `Bin ${Number(index).toLocaleString()}`;
  if (pooled) {
    const detail = document.createElement("span");
    detail.className = "huffman-pool-note";
    detail.textContent = `Pooled · ${bins.pool_reasons[rowIndex]}`;
    binCell.append(detail);
  }
  const intervalCell = document.createElement("td");
  intervalCell.textContent = `[${compactNumber(lower, 6)}, ${compactNumber(upper, 6)})`;
  const frequencyCell = document.createElement("td");
  frequencyCell.textContent = count.toLocaleString();
  const codeCell = document.createElement("td");
  const codeValue = document.createElement("code");
  codeValue.textContent = code;
  codeCell.append(codeValue);
  const lengthCell = document.createElement("td");
  lengthCell.textContent = `${codeLength} bit${codeLength === 1 ? "" : "s"}`;
  const payloadCell = document.createElement("td");
  payloadCell.textContent = (count * codeLength).toLocaleString();
  for (const cell of [binCell, intervalCell, frequencyCell, codeCell, lengthCell, payloadCell]) row.append(cell);
  row.setAttribute("aria-rowindex", String(rank + 1));
  huffmanCodeRows.append(row);
}

function renderHuffmanPreviewRows() {
  if (!huffmanData) return;
  const total = huffmanVisibleRows.length;
  const headerHeight = huffmanCodeScroll.querySelector("thead").getBoundingClientRect().height;
  const firstVisible = Math.floor(Math.max(0, huffmanCodeScroll.scrollTop - headerHeight) / HUFFMAN_ROW_HEIGHT);
  const visibleCount = Math.ceil(Math.max(0, huffmanCodeScroll.clientHeight - headerHeight) / HUFFMAN_ROW_HEIGHT);
  const start = Math.max(0, firstVisible - TABLE_OVERSCAN);
  const end = Math.min(total, firstVisible + visibleCount + TABLE_OVERSCAN);
  huffmanCodeRows.replaceChildren();
  if (total === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "loading-cell";
    cell.textContent = "No bins match this search.";
    row.append(cell);
    huffmanCodeRows.append(row);
  } else {
    appendHuffmanSpacer(start * HUFFMAN_ROW_HEIGHT);
    for (let rank = start; rank < end; rank++) appendHuffmanCodeRow(huffmanVisibleRows[rank], rank);
    appendHuffmanSpacer((total - end) * HUFFMAN_ROW_HEIGHT);
  }
  const rangeStart = total ? start + 1 : 0;
  const rangeEnd = Math.min(end, total);
  const searchLabel = huffmanCodeSearch.value.trim() ? ` · ${total.toLocaleString()} match${total === 1 ? "" : "es"}` : "";
  document.querySelector("#huffman-code-foot").textContent =
    `Rows ${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()} of ${total.toLocaleString()}${searchLabel} · each row maps one source bin to its assigned Huffman code.`;
}

function makeSvgElement(name, attributes = {}, textContent = "") {
  const element = document.createElementNS(SWEEP_SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (textContent) element.textContent = textContent;
  return element;
}

function sweepChartSpecs() {
  return [
    { id: "sweep-chart-mae", key: "mae", color: "#2d8277", format: (value) => compactNumber(value, 4) },
    { id: "sweep-chart-mse", key: "mse", color: "#d28a56", format: (value) => compactNumber(value, 4) },
    { id: "sweep-chart-bins", key: "occupied_input_bins", color: "#557fa1", format: (value) => Math.round(value).toLocaleString() },
    { id: "sweep-chart-rate", key: "average_code_bits_per_original_weight", color: "#826cb1", format: (value) => `${Number(value).toFixed(3)} bits` },
  ];
}

function drawSweepChart(svg, spec) {
  const points = binWidthSweepData.points;
  const viewWidth = 1000;
  const viewHeight = 184;
  const left = 76;
  const right = 24;
  const top = 12;
  const bottom = 34;
  const plotWidth = viewWidth - left - right;
  const plotHeight = viewHeight - top - bottom;
  const values = points.map((point) => Number(point[spec.key]));
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    const padding = minimum === 0 ? 1 : Math.abs(minimum) * 0.08;
    minimum = Math.max(0, minimum - padding);
    maximum += padding;
  } else {
    const padding = (maximum - minimum) * 0.08;
    minimum = Math.max(0, minimum - padding);
    maximum += padding;
  }
  const xFor = (index) => left + (index / Math.max(1, points.length - 1)) * plotWidth;
  const yFor = (value) => top + ((maximum - value) / (maximum - minimum)) * plotHeight;

  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${viewWidth} ${viewHeight}`);
  const grid = makeSvgElement("g", { class: "sweep-grid" });
  for (let tick = 0; tick <= 4; tick++) {
    const value = maximum - (maximum - minimum) * tick / 4;
    const y = yFor(value);
    grid.append(makeSvgElement("line", { x1: left, x2: viewWidth - right, y1: y, y2: y, class: "sweep-grid-line" }));
    grid.append(makeSvgElement("text", { x: left - 10, y: y + 3, class: "sweep-axis-label sweep-y-label", "text-anchor": "end" }, spec.format(value)));
  }
  const tickIndices = [...new Set([0, Math.round((points.length - 1) / 4), Math.round((points.length - 1) / 2), Math.round(3 * (points.length - 1) / 4), points.length - 1])];
  for (const index of tickIndices) {
    const x = xFor(index);
    grid.append(makeSvgElement("line", { x1: x, x2: x, y1: top + plotHeight, y2: top + plotHeight + 4, class: "sweep-axis-tick" }));
    grid.append(makeSvgElement("text", { x, y: viewHeight - 7, class: "sweep-axis-label", "text-anchor": "middle" }, compactNumber(points[index].bin_width, 4)));
  }
  svg.append(grid);

  const pathData = points.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(index)},${yFor(Number(point[spec.key]))}`).join(" ");
  svg.append(makeSvgElement("path", { d: pathData, class: "sweep-series-line", stroke: spec.color }));
  const seriesPoints = makeSvgElement("g", { class: "sweep-points" });
  points.forEach((point, index) => {
    seriesPoints.append(makeSvgElement("circle", {
      cx: xFor(index), cy: yFor(Number(point[spec.key])), r: 3.5,
      class: "sweep-point", "data-index": index, fill: spec.color,
    }));
  });
  svg.append(seriesPoints);
  svg.append(makeSvgElement("line", {
    x1: xFor(0), x2: xFor(0), y1: top, y2: top + plotHeight,
    class: "sweep-selected-cursor", "data-cursor": "true",
  }));
}

function updateSweepInspector(index) {
  if (!binWidthSweepData) return;
  const points = binWidthSweepData.points;
  const selected = Math.max(0, Math.min(points.length - 1, Number(index)));
  const point = points[selected];
  binWidthSweepRange.value = String(selected);
  document.querySelector("#sweep-inspector").textContent =
    `Δ ${compactNumber(point.bin_width, 5)} · MAE ${compactNumber(point.mae, 5)} · MSE ${compactNumber(point.mse, 5)} · occupied input bins ${point.occupied_input_bins.toLocaleString()} · output symbols ${point.output_symbols.toLocaleString()} · average ${point.average_code_bits_per_original_weight.toFixed(3)} bits/weight (${binWidthSweepData.settings.coding_method})`;
  for (const svg of document.querySelectorAll(".sweep-chart")) {
    for (const dot of svg.querySelectorAll(".sweep-point")) {
      const isSelected = Number(dot.dataset.index) === selected;
      dot.classList.toggle("selected", isSelected);
      dot.setAttribute("r", isSelected ? "6" : "3.5");
    }
    const cursor = svg.querySelector('[data-cursor="true"]');
    if (cursor) {
      const x = 76 + (selected / Math.max(1, points.length - 1)) * (1000 - 76 - 24);
      cursor.setAttribute("x1", String(x));
      cursor.setAttribute("x2", String(x));
    }
  }
}

function renderBinWidthSweep(data) {
  binWidthSweepData = data;
  const scope = data.dataset.scope === "__all__" ? "Full model" : data.dataset.scope;
  const method = data.settings.coding_method === "huffman" ? "Huffman" : "Unary";
  const maxLength = data.settings.max_code_length === null ? "off" : `${data.settings.max_code_length} bits`;
  document.querySelector("#bin-width-sweep-config").textContent =
    `${scope} · ${data.dataset.values.toLocaleString()} weights · ${method} · K ${data.settings.merge_rare_bins} · L ${maxLength} · ${data.points.length} widths from ${compactNumber(data.settings.minimum_bin_width, 4)} to ${compactNumber(data.settings.maximum_bin_width, 4)}`;
  binWidthSweepRange.max = String(data.points.length - 1);
  const centerIndex = data.points.reduce((best, point, index) =>
    Math.abs(Math.log(point.bin_width / data.settings.center_bin_width)) < Math.abs(Math.log(data.points[best].bin_width / data.settings.center_bin_width)) ? index : best, 0);
  for (const spec of sweepChartSpecs()) drawSweepChart(document.querySelector(`#${spec.id}`), spec);
  updateSweepInspector(centerIndex);
}

function inspectSweepPointer(event) {
  if (!binWidthSweepData) return;
  const svg = event.currentTarget;
  const bounds = svg.getBoundingClientRect();
  const viewX = (event.clientX - bounds.left) * 1000 / bounds.width;
  const plotX = Math.max(76, Math.min(1000 - 24, viewX));
  const fraction = (plotX - 76) / (1000 - 76 - 24);
  updateSweepInspector(Math.round(fraction * (binWidthSweepData.points.length - 1)));
}

async function openBinWidthSweep() {
  if (binWidthSweepLoading || binWidthSweepButton.disabled) return;
  binWidthSweepLoading = true;
  syncBinWidthSweepButton();
  binWidthSweepButton.textContent = "Calculating sweep…";
  const loading = document.querySelector("#bin-width-sweep-loading");
  const content = document.querySelector("#bin-width-sweep-content");
  loading.textContent = "Calculating the selected scope at several bin widths…";
  loading.classList.remove("sweep-loading-error");
  loading.hidden = false;
  content.hidden = true;
  if (!binWidthSweepDialog.open) binWidthSweepDialog.showModal();
  try {
    clearTimeout(requestTimer);
    const current = await loadAnalysis();
    if (!current) throw new Error("Set valid quantization settings before opening the sweep.");
    const params = new URLSearchParams({
      center_bin_width: String(current.bin_width),
      tensor: current.dataset.scope,
      merge_rare_bins: String(current.metrics.merged_rare_bin_count),
      coding_method: current.coding_method,
    });
    if (current.metrics.max_code_length !== null) params.set("max_code_length", String(current.metrics.max_code_length));
    const response = await fetch(`/api/bin-width-sweep?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not calculate the bin-width sweep.");
    renderBinWidthSweep(data);
    loading.hidden = true;
    content.hidden = false;
  } catch (error) {
    loading.textContent = error.message;
    loading.classList.add("sweep-loading-error");
  } finally {
    binWidthSweepLoading = false;
    syncBinWidthSweepButton();
    binWidthSweepButton.textContent = "Explore bin-width trade-offs";
  }
}

function bpeRows() {
  if (!bpeData) return [];
  if (bpeTableKind === "bins") {
    return bpeData.original_bins.map((bin) => ({
      search: `${bin.symbol} ${bin.frequency}`,
      cells: [
        `BIN ${bin.symbol}`,
        `[${compactNumber(bin.lower, 6)}, ${compactNumber(bin.upper, 6)})`,
        compactNumber(bin.midpoint, 6),
        bin.frequency.toLocaleString(),
      ],
    }));
  }
  if (bpeTableKind === "merges") {
    return bpeData.merges.map((merge) => ({
      search: `${merge.rank} ${merge.left_symbol} ${merge.right_symbol} ${merge.new_symbol}`,
      cells: [
        merge.rank.toLocaleString(),
        `#${merge.left_symbol}`,
        `#${merge.right_symbol}`,
        `#${merge.new_symbol}`,
        merge.pair_frequency.toLocaleString(),
        merge.merged_occurrences.toLocaleString(),
      ],
    }));
  }
  const mergeBySymbol = new Map(bpeData.merges.map((merge) => [merge.new_symbol, merge]));
  return bpeData.final_symbols.map((symbol, index) => ({
    search: `${symbol.symbol} ${symbol.frequency}`,
    cells: [
      (index + 1).toLocaleString(),
      `#${symbol.symbol}`,
      symbol.frequency.toLocaleString(),
      symbol.huffman_code,
      `${symbol.code_length} bit${symbol.code_length === 1 ? "" : "s"}`,
      mergeBySymbol.has(symbol.symbol)
        ? `#${symbol.merge_left} + #${symbol.merge_right}`
        : `bin ${symbol.symbol}`,
    ],
  }));
}

function renderBpeTable() {
  const definitions = {
    bins: { headers: ["BIN SYMBOL", "VALUE INTERVAL", "MIDPOINT", "WEIGHT COUNT"], empty: "No occupied source bins." },
    merges: { headers: ["MERGE", "LEFT SYMBOL", "RIGHT SYMBOL", "NEW SYMBOL", "PAIR FREQUENCY", "REPLACED PAIRS"], empty: "No pair met the minimum frequency." },
    symbols: { headers: ["RANK", "FINAL SYMBOL", "OCCURRENCES", "HUFFMAN CODE", "CODE LENGTH", "PAIR EXPANSION"], empty: "No final symbols." },
  };
  const definition = definitions[bpeTableKind];
  const head = document.querySelector("#bpe-table-head");
  head.replaceChildren();
  const headerRow = document.createElement("tr");
  for (const label of definition.headers) {
    const cell = document.createElement("th");
    cell.textContent = label;
    headerRow.append(cell);
  }
  head.append(headerRow);

  const query = document.querySelector("#bpe-search").value.trim().toLowerCase();
  const rows = bpeRows().filter((row) => !query || row.search.toLowerCase().includes(query));
  const pageCount = Math.max(1, Math.ceil(rows.length / BPE_PAGE_SIZE));
  bpePage = Math.min(bpePage, pageCount - 1);
  const start = bpePage * BPE_PAGE_SIZE;
  const body = document.querySelector("#bpe-table-rows");
  body.replaceChildren();
  for (const rowData of rows.slice(start, start + BPE_PAGE_SIZE)) {
    const row = document.createElement("tr");
    for (const value of rowData.cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = definition.headers.length;
    cell.className = "loading-cell";
    cell.textContent = query ? "No matching rows." : definition.empty;
    row.append(cell);
    body.append(row);
  }
  const end = Math.min(rows.length, start + BPE_PAGE_SIZE);
  document.querySelector("#bpe-page-label").textContent = rows.length
    ? `Rows ${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${rows.length.toLocaleString()}${bpeTableKind === "symbols" ? " shown (top 100)" : ""}`
    : "0 rows";
  document.querySelector("#bpe-prev-page").disabled = bpePage === 0;
  document.querySelector("#bpe-next-page").disabled = bpePage >= pageCount - 1;
}

async function openBpeView() {
  const loading = document.querySelector("#bpe-loading");
  const content = document.querySelector("#bpe-content");
  loading.textContent = "Counting bins and learning frequent ordered pairs…";
  loading.classList.remove("sweep-loading-error");
  loading.hidden = false;
  content.hidden = true;
  if (!bpeDialog.open) bpeDialog.showModal();
  await runBpeAnalysis();
}

async function runBpeAnalysis() {
  const loading = document.querySelector("#bpe-loading");
  const content = document.querySelector("#bpe-content");
  const mergeLimit = Number(bpeMergeInput.value);
  if (!Number.isInteger(mergeLimit) || mergeLimit < 0 || mergeLimit > 1000) {
    loading.textContent = "Choose a whole number of BPE merge rules from 0 to 1,000.";
    loading.classList.add("sweep-loading-error");
    loading.hidden = false;
    content.hidden = true;
    return;
  }
  bpeRunButton.disabled = true;
  bpeRunButton.textContent = "Calculating…";
  loading.textContent = "Quantizing the selected scope, applying BPE merges, and building Huffman codes…";
  loading.classList.remove("sweep-loading-error");
  loading.hidden = false;
  content.hidden = true;
  try {
    const current = await loadAnalysis();
    if (!current) throw new Error("Set valid quantization settings before running BPE + Huffman.");
    const params = new URLSearchParams({
      bin_width: String(current.bin_width),
      tensor: current.dataset.scope,
      merges: String(mergeLimit),
      min_pair_frequency: "2",
    });
    const response = await fetch(`/api/bpe-huffman-analysis?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not calculate BPE + Huffman metrics.");
    bpeData = data;
    const scopeName = data.dataset.scope === "__all__" ? "Whole model" : data.dataset.scope;
    const shape = data.dataset.shape ? ` · ${data.dataset.shape.join(" × ")}` : "";
    document.querySelector("#bpe-config").textContent = `${scopeName}${shape} · ${data.tensor_count} tensor sequence${data.tensor_count === 1 ? "" : "s"} · Δ ${data.bin_width} · ${data.merge_count_requested} requested merges`;
    document.querySelector("#bpe-weight-count").textContent = data.weight_count.toLocaleString();
    document.querySelector("#bpe-original-bin-count").textContent = data.original_occupied_bins.toLocaleString();
    document.querySelector("#bpe-merge-count").textContent = data.bpe_merge_count.toLocaleString();
    document.querySelector("#bpe-vocabulary-count").textContent = data.bpe_vocabulary_symbols.toLocaleString();
    document.querySelector("#bpe-final-symbol-count").textContent = data.final_symbol_types.toLocaleString();
    document.querySelector("#bpe-final-sequence-length").textContent = data.final_sequence_length.toLocaleString();
    document.querySelector("#bpe-mae").textContent = compactNumber(data.quantization_error.mae, 6);
    document.querySelector("#bpe-rmse").textContent = compactNumber(data.quantization_error.rmse, 6);
    document.querySelector("#bpe-max-error").textContent = compactNumber(data.quantization_error.max_abs_error, 6);
    document.querySelector("#bpe-bits-per-weight").textContent = `${data.huffman.average_bits_per_weight.toFixed(3)} bits`;
    document.querySelector("#bpe-baseline-bits-per-weight").textContent = `${data.huffman.baseline_average_bits_per_weight.toFixed(3)} bits`;
    const totalSaved = data.huffman.total_bits_saved_vs_baseline;
    document.querySelector("#bpe-total-saved").textContent = `${formatBits(Math.abs(totalSaved))} ${totalSaved >= 0 ? "saved" : "extra"}`;
    document.querySelector("#bpe-footnote").textContent = `Midpoint reconstruction expands each BPE token to its source bin IDs, so quantization error is unchanged. BPE + Huffman payload: ${formatBits(data.huffman.payload_bits)}; baseline Huffman payload: ${formatBits(data.huffman.baseline_payload_bits)}. Estimated BPE merge-rule metadata is included in the total-size comparison.`;
    bpeTableKind = "bins";
    bpePage = 0;
    document.querySelectorAll(".bpe-tab").forEach((tab) => {
      const active = tab.dataset.bpeTable === bpeTableKind;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    document.querySelector("#bpe-search").value = "";
    renderBpeTable();
    loading.hidden = true;
    content.hidden = false;
  } catch (error) {
    loading.textContent = error.message;
    loading.classList.add("sweep-loading-error");
  } finally {
    bpeRunButton.disabled = false;
    bpeRunButton.textContent = "Run BPE + Huffman";
  }
}

function renderComparison(data) {
  const current = data.metrics;
  const baseline = data.unfiltered_comparison?.metrics ?? current;
  const rows = [
    ["Output buckets", current.occupied_bins.toLocaleString(), baseline.occupied_bins.toLocaleString()],
    ["Source bins pooled by K", current.merged_rare_bin_count.toLocaleString(), baseline.merged_rare_bin_count.toLocaleString()],
    ["Additional source bins pooled by L", current.additional_length_limited_bin_count.toLocaleString(), baseline.additional_length_limited_bin_count.toLocaleString()],
    ["Maximum code length limit", current.max_code_length === null ? "Off" : `${current.max_code_length} bits`, "Off"],
    ["Longest output code", `${current.actual_max_code_length} bits`, `${baseline.actual_max_code_length} bits`],
    ["Mean absolute error", compactNumber(current.mae, 6), compactNumber(baseline.mae, 6)],
    ["Root mean squared error", compactNumber(current.rmse, 6), compactNumber(baseline.rmse, 6)],
    ["Maximum absolute error", compactNumber(current.max_abs_error, 6), compactNumber(baseline.max_abs_error, 6)],
    ["Average code bits / original weight", `${current.average_code_bits_per_original_weight.toFixed(3)} bits/weight`, `${baseline.average_code_bits_per_original_weight.toFixed(3)} bits/weight`],
    ["Entropy lower bound / weight", `${current.entropy_bits_per_weight.toFixed(3)} bits/weight`, `${baseline.entropy_bits_per_weight.toFixed(3)} bits/weight`],
    ["Code payload", formatPayloadSize(current.coding_payload_bits), formatPayloadSize(baseline.coding_payload_bits)],
    ["Code table estimate", formatPayloadSize(current.estimated_table_bits), formatPayloadSize(baseline.estimated_table_bits)],
    ["Total estimated size", `${formatBytes(current.estimated_total_bytes)} · ${formatBits(current.estimated_total_bits)}`, `${formatBytes(baseline.estimated_total_bytes)} · ${formatBits(baseline.estimated_total_bits)}`],
    ["Compared with FP32", formatRelativeSize(current.estimated_fp32_reduction_percent), formatRelativeSize(baseline.estimated_fp32_reduction_percent)],
  ];
  document.querySelector("#comparison-rows").innerHTML = rows
    .map(([label, currentValue, baselineValue]) => `<tr><td>${label}</td><td>${currentValue}</td><td>${baselineValue}</td></tr>`)
    .join("");
}

function renderTable(data) {
  activeData = data;
  frequencyScroll.scrollTop = 0;
  document.querySelector("#table-count").textContent = `${data.metrics.original_occupied_bins.toLocaleString()} BINS`;
  renderFrequencyWindow();
}

function renderFrequencyWindow() {
  if (!activeData) return;
  const bins = activeData.frequency_bins;
  const total = bins.indices.length;
  const body = document.querySelector("#frequency-rows");
  const headerHeight = frequencyScroll.querySelector("thead").getBoundingClientRect().height;
  const firstVisible = Math.floor(Math.max(0, frequencyScroll.scrollTop - headerHeight) / FREQUENCY_ROW_HEIGHT);
  const visibleCount = Math.ceil(Math.max(0, frequencyScroll.clientHeight - headerHeight) / FREQUENCY_ROW_HEIGHT);
  const start = Math.max(0, firstVisible - TABLE_OVERSCAN);
  const end = Math.min(total, firstVisible + visibleCount + TABLE_OVERSCAN);
  const html = [];
  if (start > 0) html.push(`<tr class="virtual-spacer" aria-hidden="true"><td colspan="5" style="height:${start * FREQUENCY_ROW_HEIGHT}px"></td></tr>`);
  for (let rank = start; rank < end; rank++) {
    const index = bins.indices[rank];
    const count = bins.counts[rank];
    const codeLength = bins.code_lengths[rank];
    const pooled = bins.pooled[rank];
    const lower = index * activeData.bin_width;
    const upper = (index + 1) * activeData.bin_width;
    const poolReason = bins.pool_reasons[rank];
    const status = pooled ? `<span class="merged-rep pooled-mini">POOLED · ${poolReason}</span>` : '<span class="merged-rep">own bucket</span>';
    const title = pooled ? `This source bin maps to a shared bucket (${poolReason}).` : "This bin keeps its own output bucket.";
    html.push(`<tr class="frequency-row" title="${title}"><td>${String(rank + 1).padStart(2, "0")}</td><td>[${compactNumber(lower, 5)}, ${compactNumber(upper, 5)})${status}</td><td>${count.toLocaleString()}</td><td>${bins.codes[rank]}</td><td>${(count * codeLength).toLocaleString()}</td></tr>`);
  }
  if (end < total) html.push(`<tr class="virtual-spacer" aria-hidden="true"><td colspan="5" style="height:${(total - end) * FREQUENCY_ROW_HEIGHT}px"></td></tr>`);
  body.innerHTML = html.join("");
  const rangeStart = total ? start + 1 : 0;
  document.querySelector("#table-foot").textContent = `Rows ${rangeStart.toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()} · ${activeData.frequency_order === "desc" ? "most-to-least" : "least-to-most"} frequent · POOLED marks bins sharing a code.`;
}

async function loadModelLayers() {
  setStatus("Loading model tensors…");
  try {
    const response = await fetch("/api/layers");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not load model tensors");
    modelInfo = data;
    tensorSelect.replaceChildren();

    const full = document.createElement("option");
    full.value = "__all__";
    full.textContent = `Full model · ${data.total_values.toLocaleString()} weights`;
    tensorSelect.append(full);
    for (const layer of data.layers) {
      const option = document.createElement("option");
      option.value = layer.name;
      option.textContent = `${layer.name} · ${layer.values.toLocaleString()}`;
      tensorSelect.append(option);
    }
    if (document.querySelector("#real-model-view").hidden) {
      document.querySelector("#dataset-title").textContent = "Tiny GPT-2";
      document.querySelector("#dataset-subtitle").textContent = `${data.tensor_count} tensors · ${data.total_values.toLocaleString()} weights`;
    }
    document.querySelector("#scope-summary").textContent = `All ${data.tensor_count} unique learned tensors · masks and tied aliases counted once.`;
    await loadAnalysis();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function drawHistogram() {
  if (!activeData) return;
  const bounds = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(bounds.width * dpr));
  canvas.height = Math.max(1, Math.round(bounds.height * dpr));
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  const left = 4, right = 4, top = 8, bottom = 18;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const counts = activeData.histogram.frequencies;
  const n = counts.length;
  const bucketCount = Math.max(1, Math.floor(chartWidth));
  const maxima = new Float64Array(bucketCount);
  const mergedMaxima = new Float64Array(bucketCount);
  const firstIndex = new Int32Array(bucketCount);
  const lastIndex = new Int32Array(bucketCount);
  const mergedFlags = activeData.histogram.is_merged;
  for (let i = 0; i < n; i++) {
    const x = Math.min(bucketCount - 1, Math.floor(i / n * bucketCount));
    if (mergedFlags[i]) mergedMaxima[x] = Math.max(mergedMaxima[x], counts[i]);
    else maxima[x] = Math.max(maxima[x], counts[i]);
    if (lastIndex[x] === 0) firstIndex[x] = i;
    lastIndex[x] = i + 1;
  }
  let maxCount = 0;
  for (let x = 0; x < bucketCount; x++) maxCount = Math.max(maxCount, maxima[x], mergedMaxima[x]);
  context.clearRect(0, 0, width, height);

  context.strokeStyle = "#edf1f0";
  context.lineWidth = 1;
  context.setLineDash([3, 4]);
  for (let i = 0; i <= 3; i++) {
    const y = top + chartHeight * i / 3;
    context.beginPath(); context.moveTo(left, y); context.lineTo(width - right, y); context.stroke();
  }
  context.setLineDash([]);

  for (let x = 0; x < bucketCount; x++) {
    if (maxima[x] === 0 && mergedMaxima[x] === 0) continue;
    const fullBarWidth = Math.max(1, chartWidth / bucketCount - .5);
    const bothKinds = maxima[x] > 0 && mergedMaxima[x] > 0;
    const barWidth = bothKinds ? fullBarWidth / 2 : fullBarWidth;
    const xPos = left + x * chartWidth / bucketCount;
    if (maxima[x] > 0) {
      const barHeight = Math.max(1, Math.sqrt(maxima[x] / maxCount) * chartHeight);
      context.fillStyle = "#4d9588";
      context.globalAlpha = .42 + .48 * Math.sqrt(maxima[x] / maxCount);
      context.fillRect(xPos, top + chartHeight - barHeight, barWidth, barHeight);
    }
    if (mergedMaxima[x] > 0) {
      const barHeight = Math.max(1, Math.sqrt(mergedMaxima[x] / maxCount) * chartHeight);
      context.fillStyle = "#df9662";
      context.globalAlpha = .58 + .4 * Math.sqrt(mergedMaxima[x] / maxCount);
      context.fillRect(xPos + (bothKinds ? barWidth : 0), top + chartHeight - barHeight, barWidth, barHeight);
    }
  }
  context.globalAlpha = 1;
  context.fillStyle = "#93a5a2";
  context.font = '9px "DM Mono", monospace';
  context.fillText(String(maxCount), left, top - 1);
  context.fillText("0", left, top + chartHeight - 2);
  canvas._plot = { left, top, chartWidth, chartHeight, bucketCount, firstIndex, lastIndex, counts, maxCount };
}

function showTooltip(event) {
  if (!activeData || !canvas._plot) return;
  const rect = canvas.getBoundingClientRect();
  const plot = canvas._plot;
  const x = event.clientX - rect.left - plot.left;
  const column = Math.min(plot.bucketCount - 1, Math.max(0, Math.floor(x / plot.chartWidth * plot.bucketCount)));
  const start = plot.firstIndex[column];
  const end = plot.lastIndex[column];
  if (end <= start) { tooltip.style.display = "none"; return; }
  let best = start;
  for (let i = start + 1; i < end; i++) if (plot.counts[i] > plot.counts[best]) best = i;
  const index = activeData.histogram.indices[best];
  const frequency = plot.counts[best];
  const codeLength = activeData.histogram.code_lengths[best];
  const methodName = activeData.coding_method === "huffman" ? "Huffman" : "Unary";
  if (activeData.histogram.is_merged[best]) {
    const mergedBins = activeData.histogram.merged_bin_counts[best];
    const representative = activeData.histogram.positions[best];
    tooltip.innerHTML = `<strong>Shared bucket · ${mergedBins.toLocaleString()} bins</strong><span>Mean reconstruction: ${compactNumber(representative, 6)}</span><span>Frequency: ${frequency.toLocaleString()}</span><span>${methodName} length: ${codeLength} bit${codeLength === 1 ? "" : "s"}</span>`;
  } else {
    const lower = index * activeData.bin_width;
    const upper = (index + 1) * activeData.bin_width;
    tooltip.innerHTML = `<strong>Bin ${Number(index).toLocaleString()}</strong><span>[${compactNumber(lower, 5)}, ${compactNumber(upper, 5)})</span><span>Frequency: ${frequency.toLocaleString()}</span><span>${methodName} length: ${codeLength} bit${codeLength === 1 ? "" : "s"}</span>`;
  }
  tooltip.style.display = "block";
  const xPos = Math.min(rect.width - 175, Math.max(5, event.clientX - rect.left + 12));
  const yPos = Math.max(0, event.clientY - rect.top - 15);
  tooltip.style.left = `${xPos}px`;
  tooltip.style.top = `${yPos}px`;
}

input.addEventListener("input", () => {
  const width = Number(input.value);
  if (Number.isFinite(width) && width >= 1e-9 && width <= 1) {
    slider.value = sliderFromWidth(width);
    document.querySelector("#width-multiple").textContent = `${compactNumber(width / REFERENCE_WIDTH, 4)}× reference`;
  }
  queueAnalysis();
});
rareBinCountInput.addEventListener("input", queueAnalysis);
maxCodeLengthInput.addEventListener("input", queueAnalysis);
codingMethodSelect.addEventListener("change", () => {
  syncHuffmanReviewButton();
  queueAnalysis();
});
slider.addEventListener("input", () => setWidth(widthFromSlider(slider.value), false));
tensorSelect.addEventListener("change", queueAnalysis);
frequencyOrderSelect.addEventListener("change", queueAnalysis);
huffmanReviewButton.addEventListener("click", async () => {
  if (huffmanReviewLoading || codingMethodSelect.value !== "huffman") return;
  clearTimeout(requestTimer);
  huffmanReviewLoading = true;
  syncHuffmanReviewButton();
  huffmanReviewButton.textContent = "Preparing code map…";
  try {
    const data = await loadAnalysis();
    if (!data) return;
    if (!huffmanSettingsMatch(data)) {
      setStatus("Settings changed while preparing the code map. Review the current settings again.", true);
      return;
    }
    openHuffmanPreview(data);
  } finally {
    huffmanReviewLoading = false;
    huffmanReviewButton.textContent = "Review Huffman codes";
    syncHuffmanReviewButton();
  }
});
document.querySelector("#close-huffman-dialog").addEventListener("click", () => huffmanDialog.close());
huffmanCodeSearch.addEventListener("input", filterHuffmanPreview);
huffmanCodeScroll.addEventListener("scroll", () => {
  if (huffmanRenderQueued) return;
  huffmanRenderQueued = true;
  requestAnimationFrame(() => {
    huffmanRenderQueued = false;
    renderHuffmanPreviewRows();
  });
});
huffmanDialog.addEventListener("click", (event) => {
  if (event.target === huffmanDialog) huffmanDialog.close();
});
binWidthSweepButton.addEventListener("click", openBinWidthSweep);
bpeButton.addEventListener("click", openBpeView);
document.querySelector("#close-bpe-dialog").addEventListener("click", () => bpeDialog.close());
bpeRunButton.addEventListener("click", runBpeAnalysis);
document.querySelectorAll(".bpe-tab").forEach((tab) => tab.addEventListener("click", () => {
  bpeTableKind = tab.dataset.bpeTable;
  bpePage = 0;
  document.querySelectorAll(".bpe-tab").forEach((other) => {
    const active = other === tab;
    other.classList.toggle("active", active);
    other.setAttribute("aria-selected", String(active));
  });
  renderBpeTable();
}));
document.querySelector("#bpe-search").addEventListener("input", () => {
  bpePage = 0;
  renderBpeTable();
});
document.querySelector("#bpe-prev-page").addEventListener("click", () => {
  bpePage = Math.max(0, bpePage - 1);
  renderBpeTable();
});
document.querySelector("#bpe-next-page").addEventListener("click", () => {
  bpePage += 1;
  renderBpeTable();
});
bpeDialog.addEventListener("click", (event) => {
  if (event.target === bpeDialog) bpeDialog.close();
});
document.querySelector("#close-bin-width-sweep").addEventListener("click", () => binWidthSweepDialog.close());
document.querySelector("#sweep-point-index").addEventListener("input", (event) => updateSweepInspector(event.currentTarget.value));
for (const svg of document.querySelectorAll(".sweep-chart")) svg.addEventListener("pointermove", inspectSweepPointer);
binWidthSweepDialog.addEventListener("click", (event) => {
  if (event.target === binWidthSweepDialog) binWidthSweepDialog.close();
});
frequencyScroll.addEventListener("scroll", () => {
  if (tableRenderQueued) return;
  tableRenderQueued = true;
  requestAnimationFrame(() => {
    tableRenderQueued = false;
    renderFrequencyWindow();
  });
});
canvas.addEventListener("mousemove", showTooltip);
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
window.addEventListener("resize", () => {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => {
    drawHistogram();
    renderFrequencyWindow();
    if (huffmanDialog.open) renderHuffmanPreviewRows();
    if (binWidthSweepDialog.open && binWidthSweepData) {
      for (const spec of sweepChartSpecs()) drawSweepChart(document.querySelector(`#${spec.id}`), spec);
      updateSweepInspector(binWidthSweepRange.value);
    }
  }, 80);
});
slider.value = sliderFromWidth(Number(input.value));
loadModelLayers();
