const analyzeBtn = document.getElementById("analyzeBtn");
const statusText = document.getElementById("statusText");
const textureList = document.getElementById("textureList");
const spotsBody = document.getElementById("spotsBody");
const summaryBody = document.getElementById("summaryBody");
const downloadRanking = document.getElementById("downloadRanking");
const downloadSummary = document.getElementById("downloadSummary");

const familyKeys = ["100", "110", "111"];
const inputs = Object.fromEntries(familyKeys.map((key) => [key, document.getElementById(`image${key}`)]));
const canvases = Object.fromEntries(familyKeys.map((key) => [key, document.getElementById(`canvas${key}`)]));
const contexts = Object.fromEntries(familyKeys.map((key) => [key, canvases[key].getContext("2d", { willReadFrequently: true })]));
const loadedImages = { "100": null, "110": null, "111": null };

let lastRanking = [];
let lastSummary = [];
let lastSpotRows = [];

const textureFamilies = {
  "{113}<361>": ["BCC", [1, 1, 3], [3, 6, 1], "{113} <361>"],
  "{100}<001>": ["BCC", [1, 0, 0], [0, 0, 1], "Cube"],
  "{110}<001>": ["BCC", [1, 1, 0], [0, 0, 1], "Goss"],
  "{110}<112>": ["BCC", [1, 1, 0], [1, 1, 2], "Brass"],
  "{001}<110>": ["BCC", [0, 0, 1], [1, 1, 0], "Rotated Cube"],
  "{011}<011>": ["BCC", [1, 1, 0], [1, 1, 0], "Rotated Goss"],
  "{112}<110>": ["BCC", [1, 1, 2], [1, 1, 0], "{112} <110>"],
  "{111}<110>": ["BCC", [1, 1, 1], [1, 1, 0], "{111} <110>"],
  "{013}<100>": ["BCC", [0, 1, 3], [1, 0, 0], "Cube-rd"],
  "{111}<112>": ["BCC", [1, 1, 1], [1, 1, 2], "{111} <112>"],
  "{011}<111>": ["BCC", [0, 1, 1], [1, 1, 1], "{011} <111>"],
  "{112}<111>": ["FCC", [1, 1, 2], [1, 1, 1], "Copper Texture"],
  "{011}<122>": ["FCC", [0, 1, 1], [1, 2, 2], "P Texture"],
  "{231}<346>": ["FCC", [2, 3, 1], [3, 4, 6], "S Texture"]
};

const referenceFamilies = {
  "100": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
  "110": [
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, 1, 0], [1, -1, 0], [-1, 0, 1],
    [0, -1, 1], [0, 1, -1], [1, 0, -1], [-1, 0, -1], [0, -1, -1], [-1, -1, 0]
  ],
  "111": [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
};

const allHkl = generateHklDirections();

for (const key of familyKeys) {
  inputs[key].addEventListener("change", async () => {
    const file = inputs[key].files?.[0];
    if (!file) return;

    loadedImages[key] = await loadImage(file);
    drawImageFit(key, loadedImages[key]);
    document.getElementById(`fileName${key}`).textContent = file.name;
    updateReadyState();
  });
}

analyzeBtn.addEventListener("click", () => {
  try {
    statusText.textContent = "Analyzing all three pole figures...";
    const result = analyzeAllPoleFigures();
    renderResults(result);
    statusText.textContent = `Detected ${result.spots.length} total red spot(s) across {100}, {110}, and {111}.`;
  } catch (error) {
    statusText.textContent = error.message;
  }
});

downloadRanking.addEventListener("click", () => {
  downloadCsv("Predicted_Texture_Ranking.csv", lastRanking);
});

downloadSummary.addEventListener("click", () => {
  downloadCsv("Filtered_Texture_Summary.csv", lastSummary);
});

function updateReadyState() {
  const ready = familyKeys.every((key) => loadedImages[key]);
  analyzeBtn.disabled = !ready;
  statusText.textContent = ready
    ? "Ready to analyze all three pole figures."
    : "Upload {100}, {110}, and {111} pole figures to begin.";
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function drawImageFit(familyKey, img) {
  const canvas = canvases[familyKey];
  const ctx = contexts[familyKey];
  const maxSide = 1000;
  const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
  canvas.width = Math.max(1, Math.round(img.width * scale));
  canvas.height = Math.max(1, Math.round(img.height * scale));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

function analyzeAllPoleFigures() {
  const minMrd = Number(document.getElementById("minMrd").value);
  const maxMrd = Number(document.getElementById("maxMrd").value);
  const tolerance = Number(document.getElementById("tolerance").value);
  const mrdCutoff = Number(document.getElementById("mrdCutoff").value);

  if (!Number.isFinite(minMrd) || !Number.isFinite(maxMrd) || maxMrd <= minMrd) {
    throw new Error("Maximum MRD must be greater than minimum MRD.");
  }

  const summaries = {};
  const allSpots = [];

  for (const familyKey of familyKeys) {
    const analysis = analyzeSinglePoleFigure(familyKey, minMrd, maxMrd, mrdCutoff);
    summaries[familyKey] = matchTextureFamily(analysis.spots, familyKey, tolerance);
    allSpots.push(...analysis.spots.map((spot) => ({ ...spot, familyKey })));
  }

  const combined = combineSummaries(summaries);
  const filtered = filterSummary(combined);
  const ranking = rankTextures(filtered);

  return { spots: allSpots, filtered, ranking };
}

function analyzeSinglePoleFigure(familyKey, minMrd, maxMrd, mrdCutoff) {
  drawImageFit(familyKey, loadedImages[familyKey]);

  const canvas = canvases[familyKey];
  const ctx = contexts[familyKey];
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const redMask = createRedMask(imageData);
  const contours = connectedComponents(imageData, redMask);

  if (!contours.length) {
    throw new Error(`No red pole figure spots were detected in {${familyKey}}.`);
  }

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = Math.min(centerX, centerY);

  const spots = contours
    .map((component) => makeSpot(component, centerX, centerY, radius, minMrd, maxMrd, mrdCutoff))
    .filter(Boolean)
    .sort((a, b) => a.distanceR - b.distanceR)
    .map((spot, index) => ({ ...spot, label: String.fromCharCode(65 + index) }));

  drawAnnotatedImage(familyKey, spots, centerX, centerY, radius);
  return { spots };
}

function createRedMask(imageData) {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const [h, s, v] = rgbToHsv(r, g, b);
    if (((h <= 10 || h >= 170) && s >= 70 && v >= 50) || (r > 130 && r > g * 1.35 && r > b * 1.35)) {
      mask[p] = 1;
    }
  }

  return mask;
}

function connectedComponents(imageData, mask) {
  const { data, width, height } = imageData;
  const visited = new Uint8Array(mask.length);
  const components = [];
  const minPixels = Math.max(6, Math.floor((width * height) * 0.000006));

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;

    const stack = [start];
    visited[start] = 1;
    const pixels = [];
    let sumX = 0;
    let sumY = 0;
    let sumRed = 0;

    while (stack.length) {
      const p = stack.pop();
      const x = p % width;
      const y = Math.floor(p / width);
      pixels.push(p);
      sumX += x;
      sumY += y;
      sumRed += data[p * 4];

      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = ny * width + nx;
          if (mask[ni] && !visited[ni]) {
            visited[ni] = 1;
            stack.push(ni);
          }
        }
      }
    }

    if (pixels.length >= minPixels) {
      components.push({
        count: pixels.length,
        cx: sumX / pixels.length,
        cy: sumY / pixels.length,
        meanRed: sumRed / pixels.length
      });
    }
  }

  return components;
}

function makeSpot(component, centerX, centerY, radius, minMrd, maxMrd, mrdCutoff) {
  const td = (component.cx - centerX) / radius;
  const rd = -(component.cy - centerY) / radius;
  const absRd = Math.abs(round(rd, 2));
  const absTd = Math.abs(round(td, 2));
  const distanceR = round(Math.sqrt(rd ** 2 + td ** 2), 3);

  if (distanceR > 1.08) return null;

  const denom1 = absRd ** 2 + absTd ** 2 + 1;
  const theta1 = round(degrees(Math.acos(clamp((2 * absRd) / denom1, -1, 1))), 2);
  const cosTheta2 = clamp((1 - distanceR ** 2) / (1 + distanceR ** 2), -1, 1);
  const theta2 = round(degrees(Math.acos(cosTheta2)), 2);
  const mrd = round(minMrd + (component.meanRed / 255) * (maxMrd - minMrd), 2);

  return {
    rd: round(rd, 2),
    td: round(td, 2),
    absRd,
    absTd,
    distanceR,
    theta1,
    theta2,
    mrd,
    textured: mrd > mrdCutoff
  };
}

function drawAnnotatedImage(familyKey, spots, centerX, centerY, radius) {
  drawImageFit(familyKey, loadedImages[familyKey]);
  const ctx = contexts[familyKey];
  ctx.lineWidth = 3;
  ctx.font = "bold 18px system-ui";
  ctx.textBaseline = "middle";

  for (const spot of spots) {
    const x = centerX + spot.td * radius;
    const y = centerY - spot.rd * radius;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#0f766e";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#172026";
    ctx.lineWidth = 4;
    ctx.strokeText(spot.label, x + 11, y - 11);
    ctx.fillText(spot.label, x + 11, y - 11);
  }
}

function matchTextureFamily(spots, familyKey, tolerance) {
  const referenceFamily = referenceFamilies[familyKey];
  const summary = {};

  for (const spot of spots) {
    if (!spot.textured) continue;

    const matches1 = findAllMatchesWithFamilies(spot.theta1, allHkl, referenceFamily, tolerance);
    const matches2 = findAllMatchesWithFamilies(spot.theta2, allHkl, referenceFamily, tolerance);
    const pairs = [];

    for (const hkl1 of matches1) {
      for (const hkl2 of matches2) {
        if (Math.abs(dot(hkl1, hkl2)) < 1e-6) {
          pairs.push([hkl2, hkl1]);
        }
      }
    }

    const matchedInSpot = new Set();
    for (const [vec1, vec2] of pairs) {
      const norm1 = normalizeVector(vec1);
      const norm2 = normalizeVector(vec2);

      for (const [notation, values] of Object.entries(textureFamilies)) {
        const plane = normalizeVector(values[1]);
        const direction = normalizeVector(values[2]);
        if (arraysEqual(norm1, plane) && arraysEqual(norm2, direction)) {
          const key = `${notation} -> ${values[3]}`;
          if (!summary[key]) {
            summary[key] = {
              texture_notation: notation,
              texture_name: values[3],
              spot_count_100: 0,
              spot_count_110: 0,
              spot_count_111: 0,
              total_occurrences: 0
            };
          }
          summary[key].total_occurrences += 1;
          matchedInSpot.add(key);
        }
      }
    }

    for (const key of matchedInSpot) {
      summary[key][`spot_count_${familyKey}`] += 1;
    }
  }

  return summary;
}

function combineSummaries(summaries) {
  const combined = {};
  for (const familySummary of Object.values(summaries)) {
    for (const [key, item] of Object.entries(familySummary)) {
      if (!combined[key]) {
        combined[key] = {
          texture_notation: item.texture_notation,
          texture_name: item.texture_name,
          spot_count_100: 0,
          spot_count_110: 0,
          spot_count_111: 0,
          total_occurrences: 0
        };
      }
      combined[key].spot_count_100 += item.spot_count_100;
      combined[key].spot_count_110 += item.spot_count_110;
      combined[key].spot_count_111 += item.spot_count_111;
      combined[key].total_occurrences += item.total_occurrences;
    }
  }
  return Object.values(combined);
}

function filterSummary(rows) {
  return rows.filter((row) => {
    const counts = [row.spot_count_100, row.spot_count_110, row.spot_count_111];
    const nonzero = counts.filter((value) => value > 0).length;
    return nonzero >= 2 || counts.some((value) => value >= 3);
  });
}

function rankTextures(rows) {
  if (!rows.length) return [];

  const maxOcc = Math.max(...rows.map((row) => row.total_occurrences), 1);
  const max100 = Math.max(...rows.map((row) => row.spot_count_100), 1);
  const max110 = Math.max(...rows.map((row) => row.spot_count_110), 1);
  const max111 = Math.max(...rows.map((row) => row.spot_count_111), 1);

  const scored = rows.map((row) => {
    const occ = row.total_occurrences / maxOcc;
    const sc100 = row.spot_count_100 / max100;
    const sc110 = row.spot_count_110 / max110;
    const sc111 = row.spot_count_111 / max111;
    const rawScore = 0.15 * occ + 0.24 * sc100 + 0.29 * sc110 + 0.45 * sc111 + 0.34 * (sc110 * sc111) + 0.24 * (sc100 * sc111);
    return { ...row, rawScore };
  });

  const maxScore = Math.max(...scored.map((row) => row.rawScore), 1);
  return scored
    .map((row) => ({ ...row, predicted_score: row.rawScore / maxScore }))
    .sort((a, b) => b.predicted_score - a.predicted_score);
}

function renderResults(result) {
  lastRanking = result.ranking.map((row) => ({
    Texture_Notation: row.texture_notation,
    Texture_Name: row.texture_name,
    Spot_count_100: row.spot_count_100,
    Spot_count_110: row.spot_count_110,
    Spot_count_111: row.spot_count_111,
    Total_occurrences: row.total_occurrences,
    Predicted_Score: round(row.predicted_score, 4)
  }));

  lastSummary = lastRanking.map(({ Predicted_Score, ...rest }) => rest);
  lastSpotRows = result.spots.map((spot) => ({
    Pole_Figure: `{${spot.familyKey}}`,
    Label: spot.label,
    RD: spot.rd,
    TD: spot.td,
    Distance_R: spot.distanceR,
    Theta_Model1: spot.theta1,
    Theta_Model2: spot.theta2,
    MRD: spot.mrd,
    Textured: spot.textured ? "Yes" : "No"
  }));

  downloadRanking.disabled = lastRanking.length === 0;
  downloadSummary.disabled = lastSummary.length === 0;

  textureList.classList.toggle("empty-state", result.ranking.length === 0);
  textureList.innerHTML = result.ranking.length
  ? result.ranking.slice(0, 5).map((row, index) => `
      <div class="texture-item">
        <div class="rank">${index + 1}</div>

        <div class="texture-info">
          <div class="texture-name">${escapeHtml(row.texture_name)}</div>
          <div class="texture-notation">${escapeHtml(row.texture_notation)}</div>
        </div>

      </div>
    `).join("")
  : "No filtered texture components matched these pole figures.";

  spotsBody.innerHTML = result.spots.length
    ? result.spots.map((spot) => `
      <tr>
        <td>${spot.label}</td>
        <td>{${spot.familyKey}}</td>
        <td>${spot.rd}</td>
        <td>${spot.td}</td>
        <td>${spot.distanceR}</td>
        <td>${spot.theta1}</td>
        <td>${spot.theta2}</td>
        <td>${spot.mrd}</td>
        <td class="${spot.textured ? "yes" : "no"}">${spot.textured ? "Yes" : "No"}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="9">No spots detected.</td></tr>`;

  summaryBody.innerHTML = result.ranking.length
    ? result.ranking.map((row) => `
      <tr>
        <td>${escapeHtml(row.texture_notation)}</td>
        <td>${escapeHtml(row.texture_name)}</td>
        <td>${row.spot_count_100}</td>
        <td>${row.spot_count_110}</td>
        <td>${row.spot_count_111}</td>
        <td>${row.total_occurrences}</td>
        
      </tr>
    `).join("")
    : `<tr><td colspan="7">No filtered texture components matched these pole figures.</td></tr>`;
}

function generateHklDirections() {
  const directions = [];
  for (let h = -9; h <= 9; h += 1) {
    for (let k = -9; k <= 9; k += 1) {
      for (let l = -9; l <= 9; l += 1) {
        if (!(h === 0 && k === 0 && l === 0)) directions.push([h, k, l]);
      }
    }
  }
  return directions;
}

function findAllMatchesWithFamilies(targetAngle, directions, referenceFamily, tolerance) {
  const matches = [];
  for (const direction of directions) {
    for (const ref of referenceFamily) {
      const angle = computeAngle(ref, direction);
      if (Math.abs(angle - targetAngle) <= tolerance) {
        matches.push(direction);
        break;
      }
    }
  }
  return matches;
}

function computeAngle(a, b) {
  const denom = magnitude(a) * magnitude(b);
  return degrees(Math.acos(clamp(dot(a, b) / denom, -1, 1)));
}

function normalizeVector(vec) {
  const divisor = gcd(gcd(Math.abs(vec[0]), Math.abs(vec[1])), Math.abs(vec[2])) || 1;
  return vec.map((value) => Math.abs(Math.trunc(value / divisor))).sort((a, b) => a - b);
}

function rgbToHsv(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  if (h < 0) h += 360;
  return [h / 2, max === 0 ? 0 : (delta / max) * 255, max * 255];
}

function downloadCsv(filename, rows) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(","))
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function magnitude(vec) {
  return Math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2);
}

function arraysEqual(a, b) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function gcd(a, b) {
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}

function degrees(radians) {
  return radians * 180 / Math.PI;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
