const STORAGE_TOKEN_KEY = 'landcoverai_token';

function getToken() {
  return localStorage.getItem(STORAGE_TOKEN_KEY) || null;
}

function setToken(token) {
  if (token) {
    localStorage.setItem(STORAGE_TOKEN_KEY, token);
  } else {
    localStorage.removeItem(STORAGE_TOKEN_KEY);
  }
}

function clearSession() {
  setToken(null);
}

function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return fetch(path, {
    credentials: 'same-origin',
    ...options,
    headers,
  });
}

function showToast(message, type = 'info', timeout = 3500) {
  const containerId = '_toast_container';
  let container = document.getElementById(containerId);
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.style.position = 'fixed';
    container.style.bottom = '1.5rem';
    container.style.right = '1.5rem';
    container.style.zIndex = '9999';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '0.75rem';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.textContent = message;
  toast.style.padding = '0.9rem 1.2rem';
  toast.style.borderRadius = '14px';
  toast.style.boxShadow = '0 18px 40px rgba(0,0,0,0.35)';
  toast.style.backdropFilter = 'blur(12px)';
  toast.style.background = 'rgba(20, 30, 50, 0.85)';
  toast.style.color = '#fff';
  toast.style.fontSize = '0.95rem';
  toast.style.maxWidth = '320px';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
  toast.style.transform = 'translateY(12px)';

  if (type === 'success') {
    toast.style.border = '1px solid rgba(79, 220, 137, 0.75)';
  } else if (type === 'error') {
    toast.style.border = '1px solid rgba(255, 110, 110, 0.75)';
  }

  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateY(0)';
  });

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
    setTimeout(() => toast.remove(), 250);
  }, timeout);
}

function setupNavigation() {
  const logoutBtn = document.getElementById('logoutBtn');
  if (!logoutBtn) return;

  if (getToken()) {
    logoutBtn.addEventListener('click', async event => {
      event.preventDefault();
      try {
        await apiFetch('/api/logout', { method: 'POST' });
      } catch (err) {
        // ignore
      }
      clearSession();
      window.location.href = '/login.html';
    });
  } else {
    logoutBtn.style.display = 'none';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
});
