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

    const satImg = document.getElementById('satImage');
    satImg.src = `data:image/png;base64,${data.satellite_image_base64}`;

    const predImg = document.getElementById('predImage');
    predImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;

    renderDetections(data.detections || []);
    setupDownload(data.annotated_image_base64);
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

  const center = [0, 0];
  map = L.map('map', {
    center,
    zoom: 2,
  });

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
      showToast('Prediction complete!', 'success');
    } catch (err) {
      console.error(err);
      showToast(err.message || 'Prediction failed.', 'error');
    } finally {
      setLoading(false);
    }
  });
});
