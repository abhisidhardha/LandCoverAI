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

async function getCurrentUser() {
  const token = getToken();
  if (!token) {
    return null;
  }

  try {
    const res = await apiFetch('/api/me', { method: 'GET' });
    if (!res.ok) {
      clearSession();
      return null;
    }
    const data = await res.json();
    return data?.user || null;
  } catch (_) {
    return null;
  }
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

function getRandomUserIcon() {
  const key = 'landcoverai_user_icon';
  const cached = sessionStorage.getItem(key);
  if (cached) {
    return cached;
  }

  const icons = ['AG', 'AI', 'LC', 'FM', 'RD', 'AN', 'US', 'MAP'];
  const picked = icons[Math.floor(Math.random() * icons.length)];
  sessionStorage.setItem(key, picked);
  return picked;
}

function closeUserMenus() {
  document.querySelectorAll('.user-dropdown.open').forEach(menu => {
    menu.classList.remove('open');
  });
}

function mountUserMenu(user) {
  const nav = document.querySelector('.nav');
  if (!nav) {
    return;
  }

  const existing = document.getElementById('userMenuRoot');
  if (existing) {
    existing.remove();
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'user-dropdown';
  wrapper.id = 'userMenuRoot';

  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'user-dropdown-trigger';
  trigger.setAttribute('aria-label', 'User menu');
  trigger.innerHTML = `
    <span class="user-icon">${getRandomUserIcon()}</span>
    <span class="user-name">${user?.name || t('user_menu')}</span>
    <span class="user-caret">▾</span>
  `;

  const menu = document.createElement('div');
  menu.className = 'user-dropdown-menu';
  menu.innerHTML = `
    <a href="/predictions.html">${t('nav_past_predictions')}</a>
    <button type="button" id="menuLogoutBtn">${t('nav_logout')}</button>
  `;

  trigger.addEventListener('click', event => {
    event.stopPropagation();
    const willOpen = !wrapper.classList.contains('open');
    closeUserMenus();
    if (willOpen) {
      wrapper.classList.add('open');
    }
  });

  const menuLogoutBtn = menu.querySelector('#menuLogoutBtn');
  if (menuLogoutBtn) {
    menuLogoutBtn.addEventListener('click', async event => {
      event.preventDefault();
      try {
        await apiFetch('/api/logout', { method: 'POST' });
      } catch (_) {
        // ignore
      }
      clearSession();
      window.location.href = '/login.html';
    });
  }

  wrapper.appendChild(trigger);
  wrapper.appendChild(menu);
  nav.appendChild(wrapper);
}

async function setupNavigation() {
  const navLogin = document.getElementById('navLoginLink');
  const navRegister = document.getElementById('navRegisterLink');
  const navLogout = document.getElementById('logoutBtn');
  const user = await getCurrentUser();

  if (user) {
    if (navLogin) navLogin.style.display = 'none';
    if (navRegister) navRegister.style.display = 'none';
    if (navLogout) navLogout.style.display = 'none';
    mountUserMenu(user);
    // Apply translations in case the menu was rendered after initial DOMContentLoaded
    if (typeof applyTranslations === 'function') setTimeout(applyTranslations, 0);
    return;
  }

  const existingMenu = document.getElementById('userMenuRoot');
  if (existingMenu) {
    existingMenu.remove();
  }

  if (navLogin) navLogin.style.display = '';
  if (navRegister) navRegister.style.display = '';
  if (navLogout) navLogout.style.display = 'none';
}

window.addEventListener('DOMContentLoaded', () => {
  document.addEventListener('click', () => closeUserMenus());
  setupNavigation();
});
