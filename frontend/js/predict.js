let map;
let marker;

function requireAuth() {
  const token = getToken();
  if (!token) {
    window.location.href = '/login.html';
    return false;
  }
  return true;
}

function updateCoords(lat, lon) {
  document.getElementById('lat').textContent = lat.toFixed(6);
  document.getElementById('lon').textContent = lon.toFixed(6);

  const latInput = document.getElementById('latInput');
  const lonInput = document.getElementById('lonInput');
  if (latInput) latInput.value = lat.toFixed(6);
  if (lonInput) lonInput.value = lon.toFixed(6);
}

function resetResults() {
  document.getElementById('satImage').src = '';
  document.getElementById('predImage').src = '';
  document.getElementById('downloadBtn').disabled = true;

  const uploadInput = document.getElementById('uploadInput');
  if (uploadInput) {
    uploadInput.value = '';
    document.getElementById('uploadBtn').disabled = true;
  }

  const tbody = document.querySelector('#detectionsTable tbody');
  tbody.innerHTML = '<tr><td colspan="3" class="muted">No results yet.</td></tr>';

  const profileEl = document.getElementById('landcoverProfile');
  if (profileEl) profileEl.innerHTML = '<p class="muted">Awaiting analysis...</p>';

  const recEl = document.getElementById('cropRecommendations');
  if (recEl) recEl.innerHTML = '<p class="muted">Awaiting analysis...</p>';
}

function renderDetections(detections) {
  const tbody = document.querySelector('#detectionsTable tbody');
  if (!detections?.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">No detections found.</td></tr>';
    return;
  }

  const rows = detections.map(det => {
    const row = document.createElement('tr');
    const classCell = document.createElement('td');
    classCell.textContent = det.class_name;
    const confCell = document.createElement('td');
    confCell.textContent = (det.confidence || 0).toFixed(2);
    const areaCell = document.createElement('td');
    areaCell.textContent = det.area ?? '—';

    row.appendChild(classCell);
    row.appendChild(confCell);
    row.appendChild(areaCell);
    return row;
  });

  tbody.innerHTML = '';
  rows.forEach(r => tbody.appendChild(r));
}

// ── Land Cover Profile ─────────────────────────────────────
const LANDCOVER_COLORS = {
  urban_land:  '#ffd600',
  agriculture: '#4caf50',
  rangeland:   '#ff9800',
  forest:      '#1b5e20',
  water:       '#1565c0',
  barren:      '#9e9e9e',
  unknown:     '#424242',
};

const LANDCOVER_LABELS = {
  urban_land:  'Urban',
  agriculture: 'Agriculture',
  rangeland:   'Rangeland',
  forest:      'Forest',
  water:       'Water',
  barren:      'Barren',
  unknown:     'Unknown',
};

function renderLandcoverProfile(profile) {
  const container = document.getElementById('landcoverProfile');
  if (!container || !profile) return;

  // Stacked bar
  let barHTML = '<div class="lc-bar">';
  for (const [key, pct] of Object.entries(profile)) {
    if (pct <= 0) continue;
    const color = LANDCOVER_COLORS[key] || '#888';
    const label = LANDCOVER_LABELS[key] || key;
    barHTML += `<div class="lc-bar-seg" style="width:${pct}%;background:${color}" title="${label}: ${pct}%"></div>`;
  }
  barHTML += '</div>';

  // Legend
  let legendHTML = '<div class="lc-legend">';
  for (const [key, pct] of Object.entries(profile)) {
    if (pct <= 0) continue;
    const color = LANDCOVER_COLORS[key] || '#888';
    const label = LANDCOVER_LABELS[key] || key;
    legendHTML += `<span class="lc-legend-item"><span class="lc-dot" style="background:${color}"></span>${label} <b>${pct}%</b></span>`;
  }
  legendHTML += '</div>';

  container.innerHTML = barHTML + legendHTML;
}

// ── Crop Recommendations ───────────────────────────────────
const CATEGORY_COLORS = {
  'Cereal':     { bg: '#e8f5e9', text: '#2e7d32' },
  'Pulse':      { bg: '#f1f8e9', text: '#558b2f' },
  'Oilseed':    { bg: '#fff8e1', text: '#f9a825' },
  'Fiber':      { bg: '#e3f2fd', text: '#1565c0' },
  'Sugar':      { bg: '#f3e5f5', text: '#7b1fa2' },
  'Plantation': { bg: '#e0f2f1', text: '#00695c' },
  'Spice':      { bg: '#fbe9e7', text: '#bf360c' },
  'Vegetable':  { bg: '#e0f7fa', text: '#00838f' },
  'Fruit':      { bg: '#fce4ec', text: '#c62828' },
  'Other':      { bg: '#ede7f6', text: '#512da8' },
};

function renderRecommendations(cropData) {
  const container = document.getElementById('cropRecommendations');
  if (!container || !cropData) return;

  const { recommendations, explanations } = cropData;
  if (!recommendations || recommendations.length === 0) {
    container.innerHTML = '<p class="muted">No crop recommendations for this area.</p>';
    return;
  }

  container.innerHTML = '';

  recommendations.forEach((crop, idx) => {
    const card = document.createElement('div');
    card.className = 'crop-card';
    card.style.animationDelay = `${idx * 0.05}s`;

    const catColor = CATEGORY_COLORS[crop.category] || { bg: '#f5f5f5', text: '#333' };
    const score = crop.suitability_score;
    const scoreColor = score >= 70 ? '#2e7d32' : score >= 40 ? '#f57f17' : '#c62828';
    const sciName = crop.scientific_name || '';

    // SHAP contributions (top 6 — one per land cover type)
    const shapData = explanations?.[String(crop.crop_id)] || [];
    const topShap = shapData.slice(0, 6);

    let shapHTML = '';
    if (topShap.length > 0) {
      const maxAbs = Math.max(...topShap.map(s => Math.abs(s.shap_value)), 0.01);
      shapHTML = '<div class="shap-chart">';
      topShap.forEach(s => {
        const pct = Math.min(Math.abs(s.shap_value) / maxAbs * 100, 100);
        const isPositive = s.shap_value >= 0;
        const barColor = isPositive ? '#2e7d32' : '#c62828';
        const featureLabel = s.feature.replace('pct_', '').replace('_', ' ');
        const obsVal = s.value !== undefined ? ` (${s.value}%)` : '';
        shapHTML += `
          <div class="shap-row">
            <span class="shap-label">${featureLabel}${obsVal}</span>
            <div class="shap-bar-track">
              <div class="shap-bar-fill" style="width:${pct}%;background:${barColor}"></div>
            </div>
            <span class="shap-val" style="color:${barColor}">${isPositive ? '+' : ''}${s.shap_value.toFixed(2)}</span>
          </div>`;
      });
      shapHTML += '</div>';
    }

    // Build agricultural details HTML
    const gc = crop.growing_conditions || {};
    let agriHTML = '';
    if (crop.explanation || gc.soil_type) {
      agriHTML = '<div class="agri-details-content">';
      if (crop.explanation) {
        agriHTML += `<div class="agri-explain"><strong>Why this crop?</strong> ${crop.explanation}</div>`;
      }
      if (gc.soil_type || gc.temperature_range || gc.annual_rainfall || gc.soil_ph || gc.growing_season) {
        agriHTML += '<table class="agri-table"><tbody>';
        if (gc.soil_type)          agriHTML += `<tr><td class="agri-lbl">🌱 Soil</td><td>${gc.soil_type}</td></tr>`;
        if (gc.temperature_range)  agriHTML += `<tr><td class="agri-lbl">🌡️ Temperature</td><td>${gc.temperature_range}</td></tr>`;
        if (gc.annual_rainfall)    agriHTML += `<tr><td class="agri-lbl">🌧️ Rainfall</td><td>${gc.annual_rainfall}</td></tr>`;
        if (gc.soil_ph)            agriHTML += `<tr><td class="agri-lbl">⚗️ Soil pH</td><td>${gc.soil_ph}</td></tr>`;
        if (gc.growing_season)     agriHTML += `<tr><td class="agri-lbl">📅 Season</td><td>${gc.growing_season}</td></tr>`;
        agriHTML += '</tbody></table>';
      }
      if (crop.fertilizers) {
        agriHTML += `<div class="agri-field"><span class="agri-tag">💊 Fertilizers</span> ${crop.fertilizers}</div>`;
      }
      if (crop.best_regions) {
        agriHTML += `<div class="agri-field"><span class="agri-tag">📍 Best Regions</span> ${crop.best_regions}</div>`;
      }
      if (crop.key_practices) {
        agriHTML += `<div class="agri-field"><span class="agri-tag">🔧 Key Practices</span> ${crop.key_practices}</div>`;
      }
      agriHTML += '</div>';
    }

    card.innerHTML = `
      <div class="crop-card-header">
        <div class="crop-rank">#${idx + 1}</div>
        <div class="crop-info">
          <div class="crop-name">${crop.name}</div>
          ${sciName ? `<div class="crop-sci-name">${sciName}</div>` : ''}
          <div class="crop-badges">
            <span class="crop-cat-badge" style="background:${catColor.bg};color:${catColor.text}">${crop.category}</span>
          </div>
        </div>
      </div>
      <div class="crop-score">
        <div class="crop-score-header">
          <span>Suitability</span>
          <span style="font-weight:700;color:${scoreColor}">${score.toFixed(1)}%</span>
        </div>
        <div class="crop-score-track">
          <div class="crop-score-fill" style="width:${score}%;background:${scoreColor}"></div>
        </div>
      </div>
      ${agriHTML ? `<details class="agri-details"><summary>📋 Agricultural Details</summary>${agriHTML}</details>` : ''}
      ${shapHTML ? `<details class="shap-details"><summary>🔍 SHAP Analysis</summary>${shapHTML}</details>` : ''}
    `;

    container.appendChild(card);
  });
}

function setupDownload(imageBase64) {
  const downloadBtn = document.getElementById('downloadBtn');
  if (!imageBase64) {
    downloadBtn.disabled = true;
    return;
  }

  downloadBtn.disabled = false;
  downloadBtn.onclick = () => {
    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${imageBase64}`;
    link.download = 'predictions.jpg';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };
}

function encodeFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setLoading(loading) {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  overlay.classList.toggle('visible', loading);
}

async function runPrediction(lat, lon, radius_m) {
  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  setLoading(true);

  try {
    const resp = await apiFetch('/api/predict/coordinates', {
      method: 'POST',
      body: JSON.stringify({ lat, lon, radius_m, size: 512 }),
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    // Guard: ensure we actually received prediction data
    if (data.rejected || !data.annotated_image_base64) {
      throw new Error(data.error || 'Prediction returned no results.');
    }

    const satImg = document.getElementById('satImage');
    satImg.src = `data:image/png;base64,${data.satellite_image_base64}`;

    const predImg = document.getElementById('predImage');
    predImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;

    renderDetections(data.detections || []);
    setupDownload(data.annotated_image_base64);

    // Crop recommendations
    if (data.crop_recommendations) {
      renderLandcoverProfile(data.crop_recommendations.landcover_profile);
      renderRecommendations(data.crop_recommendations);
    }

    showToast('Prediction complete!', 'success');
  } catch (err) {
    console.error(err);
    showToast(err.message || 'Prediction failed.', 'error');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
    setLoading(false);
  }
}

function initMap() {
  if (typeof L === 'undefined') {
    showToast('Leaflet failed to load; map cannot be initialized.', 'error');
    return;
  }

  // Fix missing marker images when using unpkg
  L.Icon.Default.imagePath = 'https://unpkg.com/leaflet@1.9.4/dist/images/';

  const defaultCenter = [22.9074, 79.1469]; // Central India fallback
  map = L.map('map', {
    center: defaultCenter,
    zoom: 5,
  });

  // 1. Try HTML5 Geolocation
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        map.setView([latitude, longitude], 14); // Tight zoom for actual GPS
      },
      (error) => {
        console.warn('Geolocation denied or unavailable. Trying IP fallback.', error);
        fetchIpLocation();
      },
      { timeout: 6000 }
    );
  } else {
    fetchIpLocation();
  }

  // 2. Fallback to IP Geolocation
  function fetchIpLocation() {
    fetch('https://ipapi.co/json/')
      .then(res => res.json())
      .then(data => {
        if (data && data.latitude && data.longitude) {
          map.setView([data.latitude, data.longitude], 10); // Broader zoom for region/city
        }
      })
      .catch(err => console.error('IP location failed:', err));
  }

  L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      attribution:
        'Tiles &copy; <a href="https://www.esri.com">Esri</a> | ' +
        'Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
      maxZoom: 19,
    }
  ).addTo(map);

  // Add roads and labels (Hybrid Map)
  L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    {
      maxZoom: 19,
    }
  ).addTo(map);

  map.on('click', e => {
    const [lat, lon] = [e.latlng.lat, e.latlng.lng];
    if (marker) {
      marker.setLatLng(e.latlng);
    } else {
      marker = L.marker(e.latlng).addTo(map);
    }

    updateCoords(lat, lon);
    document.getElementById('analyzeBtn').disabled = false;
    resetResults();
  });
}

window.addEventListener('DOMContentLoaded', () => {
  console.log('predict.js loaded');
  try {
    if (!requireAuth()) {
      return;
    }

    initMap();
  } catch (err) {
    console.error('predict.js error', err);
    showToast('An error occurred initializing the map. See console.', 'error');
  }

  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.addEventListener('click', () => {
    if (!marker) {
      showToast('Click on the map to choose a location.', 'error');
      return;
    }
    const { lat, lng } = marker.getLatLng();
    const radiusStr = document.getElementById('rangeSelect').value;
    runPrediction(lat, lng, parseFloat(radiusStr));
  });

  const gotoBtn = document.getElementById('gotoBtn');
  const latInput = document.getElementById('latInput');
  const lonInput = document.getElementById('lonInput');

  gotoBtn.addEventListener('click', () => {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);

    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      showToast('Please enter valid latitude and longitude.', 'error');
      return;
    }

    map.setView([lat, lon], Math.max(3, map.getZoom()));
    const latlng = { lat, lng: lon };

    if (marker) {
      marker.setLatLng(latlng);
    } else {
      marker = L.marker(latlng).addTo(map);
    }

    updateCoords(lat, lon);
    document.getElementById('analyzeBtn').disabled = false;
    resetResults();
  });

  const uploadInput = document.getElementById('uploadInput');
  const uploadBtn = document.getElementById('uploadBtn');

  uploadInput.addEventListener('change', () => {
    uploadBtn.disabled = uploadInput.files.length === 0;
  });

  uploadBtn.addEventListener('click', async () => {
    if (!uploadInput.files.length) {
      showToast('Please select an image to upload.', 'error');
      return;
    }

    const file = uploadInput.files[0];
    resetResults();

    try {
      setLoading(true);
      const imageBase64 = await encodeFileAsBase64(file);
      const resp = await apiFetch('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ image_base64: imageBase64 }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || 'Prediction failed.');
      }

      document.getElementById('satImage').src = URL.createObjectURL(file);
      const predImg = document.getElementById('predImage');
      predImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;

      renderDetections(data.detections || []);
      setupDownload(data.annotated_image_base64);

      // Crop recommendations
      if (data.crop_recommendations) {
        renderLandcoverProfile(data.crop_recommendations.landcover_profile);
        renderRecommendations(data.crop_recommendations);
      }

      showToast('Prediction complete!', 'success');
    } catch (err) {
      console.error(err);
      showToast(err.message || 'Prediction failed.', 'error');
    } finally {
      setLoading(false);
    }
  });
});
