function formatDateTime(value) {
  if (!value) return 'Unknown time';
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  return dt.toLocaleString();
}

function renderImageSlot(label, src) {
  if (!src) {
    return `<div class="history-image-slot"><span class="muted">No ${label} image</span></div>`;
  }
  return `<div class="history-image-slot"><img src="${src}" alt="${label}" /></div>`;
}

function renderSummaryChips(results) {
  const summary = results?.summary || {};
  const chips = Object.entries(summary)
    .filter(([, count]) => Number(count) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 6)
    .map(([name, count]) => `<span class="history-chip">${name.replaceAll('_', ' ')}: ${count}</span>`);

  if (!chips.length) {
    return '<span class="muted">No class summary available</span>';
  }
  return chips.join('');
}

function renderPredictionCard(item) {
  const rec = item?.results?.crop_recommendations?.recommendations?.[0];
  const topRecommendation = rec
    ? `<span class="history-chip">${t('top_crop')}: ${rec.name} (${Number(rec.suitability_score || 0).toFixed(1)}%)</span>`
    : '';

  return `
    <article class="history-card">
      <div class="history-meta">
        <strong>${t('prediction_num')} #${item.id}</strong>
        <span class="stamp">${formatDateTime(item.created_at)}</span>
      </div>

      <div class="history-images">
        ${renderImageSlot('original', item.original_img_path)}
        ${renderImageSlot('annotated', item.annotated_img_path)}
      </div>

      <div class="history-summary">
        ${renderSummaryChips(item.results)}
        ${topRecommendation}
      </div>
    </article>
  `;
}

async function requireHistoryAuth() {
  const user = await getCurrentUser();
  if (!user) {
    clearSession();
    window.location.href = '/login.html';
    return null;
  }
  return user;
}

async function loadHistory() {
  const subtitle = document.getElementById('historySubtitle');
  const grid = document.getElementById('historyGrid');

  const user = await requireHistoryAuth();
  if (!user) return;

  subtitle.textContent = `${t('history_for')} ${user.name}.`;

  try {
    const res = await apiFetch('/api/predictions', { method: 'GET' });
    if (res.status === 401) {
      clearSession();
      window.location.href = '/login.html';
      return;
    }

    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    const rawBody = await res.text();
    let data = null;

    if (contentType.includes('application/json')) {
      data = rawBody ? JSON.parse(rawBody) : {};
    } else {
      throw new Error(t('history_error'));
    }

    if (!res.ok) {
      throw new Error(data?.error || t('history_error'));
    }

    const items = data?.predictions || [];
    if (!items.length) {
      grid.innerHTML = `
        <article class="history-card">
          <h3 style="margin-top:0;color:var(--primary);">${t('no_preds_title')}</h3>
          <p class="muted">${t('no_preds_desc')}</p>
          <a class="btn btn-primary" href="/predict.html">${t('btn_run_first')}</a>
        </article>
      `;
      return;
    }

    grid.innerHTML = items.map(renderPredictionCard).join('');
  } catch (err) {
    console.error(err);
    subtitle.textContent = 'Unable to load prediction history right now.';
    grid.innerHTML = `
      <article class="history-card">
        <p class="muted">${err.message || 'Something went wrong while loading your history.'}</p>
      </article>
    `;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadHistory();
});
