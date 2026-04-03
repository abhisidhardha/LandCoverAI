let map;
let marker;

async function requireAuth() {
  const user = await getCurrentUser();
  if (!user) {
    clearSession();
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
  tbody.innerHTML = `<tr><td colspan="3" class="muted">${t('no_results') || 'No results yet.'}</td></tr>`;

  const profileEl = document.getElementById('landcoverProfile');
  if (profileEl) profileEl.innerHTML = `<p class="muted">${t('awaiting') || 'Awaiting analysis...'}</p>`;

  const recEl = document.getElementById('cropRecommendations');
  if (recEl) recEl.innerHTML = `<p class="muted">${t('awaiting') || 'Awaiting analysis...'}</p>`;

  const overviewEl = document.getElementById('cropOverview');
  if (overviewEl) overviewEl.innerHTML = `<p class="muted">${t('awaiting') || 'Awaiting analysis...'}</p>`;
}

function renderDetections(detections) {
  const tbody = document.querySelector('#detectionsTable tbody');
  if (!detections?.length) {
    tbody.innerHTML = `<tr><td colspan="3" class="muted">${t('no_detections') || 'No detections found.'}</td></tr>`;
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

const getLandCoverLabel = (key) => t('lc_' + key) || LANDCOVER_LABELS[key] || key;

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

  let legendHTML = '<div class="lc-legend">';
  for (const [key, pct] of Object.entries(profile)) {
    if (pct <= 0) continue;
    const color = LANDCOVER_COLORS[key] || '#888';
    const label = getLandCoverLabel(key);
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

function renderCropOverview(cropData) {
  const container = document.getElementById('cropOverview');
  if (!container || !cropData) return;

  const recommendations = cropData.recommendations || [];
  if (!recommendations.length) {
    container.innerHTML = `<p class="muted">${t('no_crop_recs') || 'No crop recommendations for this area.'}</p>`;
    return;
  }

  const rows = [...recommendations]
    .sort((a, b) => (b.suitability_score || 0) - (a.suitability_score || 0))
    .map((crop, idx) => ({ crop, rank: idx + 1 }));

  const tableRows = rows.map(({ crop, rank }) => {
    const tier = crop.risk_tier || (rank <= 5 ? 'Best Fit' : rank <= 10 ? 'Good Alternative' : 'Worth Exploring');
    const score = (crop.suitability_score || 0).toFixed(1);
    return `
      <tr>
        <td>#${rank}</td>
        <td><strong>${crop.name || 'Unknown'}</strong></td>
        <td>${crop.category || 'Other'}</td>
        <td>${tier}</td>
        <td class="crop-overview-score">${score}%</td>
      </tr>`;
  }).join('');

  const cards = rows.map(({ crop, rank }) => {
    const tier = crop.risk_tier || (rank <= 5 ? 'Best Fit' : rank <= 10 ? 'Good Alternative' : 'Worth Exploring');
    const score = (crop.suitability_score || 0).toFixed(1);
    return `
      <article class="crop-overview-card">
        <div class="crop-overview-card-head">
          <span class="crop-overview-rank">#${rank}</span>
          <span class="crop-overview-score">${score}%</span>
        </div>
        <h4>${crop.name || 'Unknown'}</h4>
        <p><strong>${t('th_class') || 'Category'}:</strong> ${crop.category || 'Other'}</p>
        <p><strong>${t('tier') || 'Tier'}:</strong> ${tier}</p>
      </article>`;
  }).join('');

  container.innerHTML = `
    <div class="crop-overview-table-wrap" style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">
      <table class="table crop-overview-table">
        <thead>
          <tr>
            <th>${t('rank') || 'Rank'}</th>
            <th>${t('crop') || 'Crop'}</th>
            <th>${t('th_class') || 'Category'}</th>
            <th>${t('tier') || 'Tier'}</th>
            <th>${t('suitability') || 'Suitability'}</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
    <div class="crop-overview-cards">${cards}</div>
  `;
}

function renderRecommendations(cropData) {
  const container = document.getElementById('cropRecommendations');
  if (!container || !cropData) return;

  const { recommendations, explanations, terrain_classification } = cropData;
  if (!recommendations || recommendations.length === 0) {
    container.innerHTML = `<p class="muted">${t('no_crop_recs') || 'No crop recommendations for this area.'}</p>`;
    return;
  }

  container.innerHTML = '';

  // ── Terrain Classification Banner ──────────────────────────────────────
  if (terrain_classification && terrain_classification.name) {
    const terrainBanner = document.createElement('div');
    terrainBanner.className = 'terrain-banner';
    terrainBanner.innerHTML = `
      <div style="display:flex;align-items:center;gap:12px;padding:16px 20px;background:linear-gradient(135deg,#1a237e,#283593);border-radius:12px;color:white;margin-bottom:20px;box-shadow:0 4px 15px rgba(26,35,126,0.3);">
        <div>
          <div style="font-size:1.1rem;font-weight:700;letter-spacing:0.5px;">Detected Terrain: ${terrain_classification.name}</div>
          <div style="font-size:0.85rem;opacity:0.85;margin-top:2px;">${terrain_classification.description || ''}</div>
        </div>
      </div>
    `;
    container.appendChild(terrainBanner);
  }

  // ── Risk Tier Configuration ────────────────────────────────────────────
  const TIER_CONFIG = {
    'Best Fit': {
      icon: '', color: '#2e7d32', bg: 'var(--primary-light)',
      border: '#2e7d32', label: 'Best Fit — High Confidence Crops',
      desc: 'These crops are the best match for your terrain. High probability of good yield.'
    },
    'Good Alternative': {
      icon: '', color: '#1565c0', bg: 'rgba(21, 101, 192, 0.1)',
      border: '#1565c0', label: 'Good Alternatives — Solid Choices',
      desc: 'Strong candidates with slightly different requirements. Worth considering for diversification.'
    },
    'Worth Exploring': {
      icon: '', color: '#e65100', bg: 'rgba(230, 81, 0, 0.1)',
      border: '#e65100', label: 'Worth Exploring — Unique Opportunities',
      desc: 'Lower suitability but unique crops that may yield well with targeted interventions.'
    },
  };

  // ── Group by tier ──────────────────────────────────────────────────────
  const tiers = {};
  recommendations.forEach((crop, idx) => {
    const tier = crop.risk_tier || (idx < 5 ? 'Best Fit' : idx < 10 ? 'Good Alternative' : 'Worth Exploring');
    if (!tiers[tier]) tiers[tier] = [];
    tiers[tier].push({ crop, globalIdx: idx });
  });

  // ── Render each tier ──────────────────────────────────────────────────
  let globalRank = 0;
  for (const tierName of ['Best Fit', 'Good Alternative', 'Worth Exploring']) {
    const tierCrops = tiers[tierName];
    if (!tierCrops || tierCrops.length === 0) continue;

    const config = TIER_CONFIG[tierName] || TIER_CONFIG['Best Fit'];

    // Tier header
    const tierHeader = document.createElement('div');
    tierHeader.className = 'tier-header';
    tierHeader.innerHTML = `
      <div style="display:flex;align-items:center;gap:12px;padding:14px 18px;background:${config.bg};border-left:4px solid ${config.border};border-radius:10px;margin:24px 0 14px 0;">
        <div>
          <div style="font-size:1rem;font-weight:700;color:${config.color};">${config.label}</div>
          <div style="font-size:0.8rem;color:#555;margin-top:2px;">${config.desc}</div>
        </div>
      </div>
    `;
    container.appendChild(tierHeader);

    // Render crop cards in this tier
    tierCrops.forEach(({ crop }) => {
      globalRank++;
      const card = document.createElement('div');
      card.className = 'crop-card';
      card.style.animationDelay = `${globalRank * 0.04}s`;

      const catColor = CATEGORY_COLORS[crop.category] || { bg: '#f5f5f5', text: '#333' };
      const score = crop.suitability_score;
      const scoreColor = score >= 70 ? '#2e7d32' : score >= 40 ? '#f57f17' : '#c62828';
      const sciName = crop.scientific_name || '';

      // SHAP contributions
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

      // Agricultural details HTML
      const gc = crop.growing_conditions || {};
      let agriHTML = '';
      if (crop.explanation || gc.soil_type) {
        agriHTML = '<div class="agri-details-content">';
        if (crop.explanation) {
          agriHTML += `<div class="agri-explain"><strong>${t('why_crop') || 'Why this crop?'}</strong> ${crop.explanation}</div>`;
        }
        if (gc.soil_type || gc.temperature_range || gc.annual_rainfall || gc.soil_ph || gc.growing_season) {
          agriHTML += '<table class="agri-table"><tbody>';
          if (gc.soil_type)          agriHTML += `<tr><td class="agri-lbl">${t('soil_label') || 'Soil'}</td><td>${gc.soil_type}</td></tr>`;
          if (gc.temperature_range)  agriHTML += `<tr><td class="agri-lbl">${t('temp_label') || 'Temperature'}</td><td>${gc.temperature_range}</td></tr>`;
          if (gc.annual_rainfall)    agriHTML += `<tr><td class="agri-lbl">${t('rain_label') || 'Rainfall'}</td><td>${gc.annual_rainfall}</td></tr>`;
          if (gc.soil_ph)            agriHTML += `<tr><td class="agri-lbl">${t('ph_label') || 'Soil pH'}</td><td>${gc.soil_ph}</td></tr>`;
          if (gc.growing_season)     agriHTML += `<tr><td class="agri-lbl">${t('season_label') || 'Season'}</td><td>${gc.growing_season}</td></tr>`;
          agriHTML += '</tbody></table>';
        }
        if (crop.fertilizers) {
          agriHTML += `<div class="agri-field"><span class="agri-tag">${t('fert_label') || 'Fertilizers'}</span> ${crop.fertilizers}</div>`;
        }
        if (crop.best_regions) {
          agriHTML += `<div class="agri-field"><span class="agri-tag">${t('region_label') || 'Best Regions'}</span> ${crop.best_regions}</div>`;
        }
        if (crop.key_practices) {
          agriHTML += `<div class="agri-field"><span class="agri-tag">${t('practice_label') || 'Key Practices'}</span> ${crop.key_practices}</div>`;
        }
        if (crop.season_bonus) {
          agriHTML += `<div class="agri-field" style="color:#2e7d32;font-weight:600;"><span class="agri-tag">${t('season_bonus') || 'Season Bonus'}</span> ${t('season_bonus_text') || 'In-Season Crop (+10 Suitability)'}</div>`;
        }
        if (crop.planting_window) {
          const pw = crop.planting_window;
          agriHTML += `<div class="agri-field"><span class="agri-tag">${t('planting_window') || 'Planting Window'}</span> ${t('sow_label') || 'Sow'}: ${pw.sow} | ${t('harvest_label') || 'Harvest'}: ${pw.harvest} <br><small class="muted">${pw.conditions||''}</small></div>`;
        }
        if (crop.rotation_benefit) {
          agriHTML += `<div class="agri-field" style="color:#2e7d32;font-weight:600;"><span class="agri-tag">${t('rotation_synergy') || 'Rotation Synergy'}</span> ${crop.rotation_benefit}</div>`;
        }
        if (crop.rotation_warning) {
          agriHTML += `<div class="agri-field" style="color:#c62828;font-weight:600;"><span class="agri-tag">${t('rotation_warning') || 'Rotation Warning'}</span> ${crop.rotation_warning}</div>`;
        }
        agriHTML += '</div>';
      }

      // Counterfactuals HTML
      let cfHTML = '';
      if (crop.counterfactuals && crop.counterfactuals.length > 0) {
        cfHTML = '<div class="cf-container" style="display:flex;gap:10px;margin-top:10px;">';
        crop.counterfactuals.forEach(cf => {
          cfHTML += `
            <div style="flex:1;background:rgba(0,0,0,0.03);padding:10px;border-radius:6px;border-left:3px solid ${cf.feasibility==='High'?'#2e7d32':'#f57f17'}">
              <div style="font-weight:600;font-size:0.9rem;">${cf.scenario}</div>
              <div style="color:var(--text-color);font-size:0.85rem;">${t('change_label') || 'Change'}: <strong>${cf.required_change}</strong></div>
              <div style="color:var(--text-color);font-size:0.85rem;">${t('new_score') || 'New Score'}: <strong>${cf.projected_score}%</strong></div>
            </div>
          `;
        });
        cfHTML += '</div>';
      }

      // Risk / Confidence Badge
      const risk = crop.prediction_risk || 'Unknown';
      const riskColor = risk === 'Low' ? '#2e7d32' : risk === 'Moderate' ? '#f57f17' : '#c62828';
      const ci = crop.confidence_interval ? ` (${crop.confidence_interval[0]} - ${crop.confidence_interval[1]}%)` : '';

      // Tier badge
      const tierBadgeColor = config.color;
      const tierBadgeBg = config.border + '18';

      card.innerHTML = `
        <div style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 1.5rem; align-items: center; justify-content: space-between;">
          <div class="crop-card-header" style="margin-bottom: 0; min-width: 280px; flex: 1;">
            <div class="crop-rank">#${globalRank}</div>
            <div class="crop-info">
              <div class="crop-name">${crop.name}</div>
              ${sciName ? `<div class="crop-sci-name">${sciName}</div>` : ''}
              <div class="crop-badges">
                <span class="crop-cat-badge" style="background:${catColor.bg};color:${catColor.text}">${t('opt_' + crop.category.toLowerCase()) || crop.category}</span>
                <span class="crop-cat-badge" style="background:${riskColor};color:white;font-weight:600;" title="Monte Carlo 90% CI${ci}">${t('risk_label') || 'Risk'}: ${risk}</span>
                <span class="crop-cat-badge" style="background:${tierBadgeBg};color:${tierBadgeColor};font-weight:600;border:1px solid ${tierBadgeColor}30;">${tierName}</span>
              </div>
            </div>
          </div>
          <div class="crop-score" style="margin-bottom: 0; min-width: 200px; max-width: 380px; flex: 1;">
            <div class="crop-score-header">
              <span>${t('suitability') || 'Suitability'}</span>
              <span style="font-weight:700;color:${scoreColor}">${score.toFixed(1)}%</span>
            </div>
            <div class="crop-score-track">
              <div class="crop-score-fill" style="width:${score}%;background:${scoreColor}"></div>
            </div>
          </div>
        </div>
        <div class="crop-card-body" style="background: rgba(0,0,0,0.03); padding: 1.25rem; border-radius: 12px; border: 1px solid var(--border); margin-top: 1.25rem; display: flex; flex-direction: column; gap: 0.8rem;">
          ${agriHTML ? `<details class="agri-details" open><summary>${t('agri_details') || 'Agricultural Details'}</summary>${agriHTML}</details>` : ''}
          ${crop.visual_explanation ? `<details class="shap-details" open><summary>${t('visual_match') || 'Advanced Visual Match'}</summary><img src="${crop.visual_explanation}" alt="Visual Explanation" style="width:100%;border-radius:6px;margin-top:10px;"></details>` : ''}
          ${cfHTML ? `<details class="shap-details"><summary>${t('opt_strategies') || 'Optimization Strategies'}</summary>${cfHTML}</details>` : ''}
          ${shapHTML ? `<details class="shap-details"><summary>${t('raw_shap') || 'Raw SHAP Data'}</summary>${shapHTML}</details>` : ''}
        </div>
      `;

      container.appendChild(card);
    });
  }
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
  analyzeBtn.textContent = '...';
  setLoading(true);

  try {
    const previous_crop_id = document.getElementById('prevCropSelect')?.value || null;
    const resp = await apiFetch('/api/predict/coordinates', {
      method: 'POST',
      body: JSON.stringify({ lat, lon, radius_m, size: 512, previous_crop_id }),
    });

    const data = await resp.json();
    if (resp.status === 401) {
      clearSession();
      window.location.href = '/login.html';
      return;
    }
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
      renderCropOverview(data.crop_recommendations);
      renderRecommendations(data.crop_recommendations);
    }

    showToast(t('prediction_complete') || 'Prediction complete!', 'success');
  } catch (err) {
    console.error(err);
    showToast(err.message || t('prediction_failed') || 'Prediction failed.', 'error');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = t('analyze_btn') || 'Analyze';
    setLoading(false);
  }
}

function initMap() {
  if (typeof L === 'undefined') {
    showToast(t('leaflet_fail') || 'Leaflet failed to load; map cannot be initialized.', 'error');
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
  (async () => {
    try {
      if (!(await requireAuth())) {
        return;
      }
      
      // Apply translations early for dynamic strings and map setup strings if any
      if (typeof applyTranslations === 'function') applyTranslations();

      initMap();
    } catch (err) {
      console.error('predict.js error', err);
      showToast('An error occurred initializing the map. See console.', 'error');
      return;
    }
  })();

  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.addEventListener('click', () => {
    if (!marker) {
      showToast(t('click_map') || 'Click on the map to choose a location.', 'error');
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
      showToast(t('valid_coords') || 'Please enter valid latitude and longitude.', 'error');
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
      showToast(t('select_image') || 'Please select an image to upload.', 'error');
      return;
    }

    const file = uploadInput.files[0];
    resetResults();

    try {
      setLoading(true);
      const previous_crop_id = document.getElementById('prevCropSelect')?.value || null;
      const imageBase64 = await encodeFileAsBase64(file);
      const resp = await apiFetch('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ image_base64: imageBase64, previous_crop_id }),
      });

      const data = await resp.json();
      if (resp.status === 401) {
        clearSession();
        window.location.href = '/login.html';
        return;
      }
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
        renderCropOverview(data.crop_recommendations);
        renderRecommendations(data.crop_recommendations);
      }

      showToast(t('prediction_complete') || 'Prediction complete!', 'success');
    } catch (err) {
      console.error(err);
      showToast(err.message || t('prediction_failed') || 'Prediction failed.', 'error');
    } finally {
      setLoading(false);
    }
  });
});
