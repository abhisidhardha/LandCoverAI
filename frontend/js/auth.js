function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validatePayload(form, payload) {
  if (form.id === 'registerForm') {
    if (!payload.name || payload.name.length < 2) {
      showToast(t('name_min'), 'error');
      return false;
    }
    if (!isValidEmail(payload.email || '')) {
      showToast(t('email_invalid'), 'error');
      return false;
    }
    if (!payload.password || payload.password.length < 8) {
      showToast(t('pw_min'), 'error');
      return false;
    }
    if (payload.password !== payload.confirm) {
      showToast(t('pw_mismatch'), 'error');
      return false;
    }
    delete payload.confirm;
    return true;
  }

  if (form.id === 'loginForm') {
    if (!isValidEmail(payload.email || '')) {
      showToast(t('email_invalid'), 'error');
      return false;
    }
    if (!payload.password) {
      showToast(t('pw_min'), 'error');
      return false;
    }
  }

  return true;
}

async function handleFormSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const url = form.dataset.endpoint;

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = value.trim();
  });

  if (!validatePayload(form, payload)) {
    return;
  }

  try {
    const res = await apiFetch(url, {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || t('server_error'), 'error');
      return;
    }

    if (form.id === 'loginForm') {
      if (data.token) {
        setToken(data.token);
        showToast(t('login_success'), 'success');
        window.location.href = '/predict.html';
        return;
      }
    }

    if (form.id === 'registerForm') {
      showToast(t('register_success'), 'success');
      window.location.href = '/login.html';
      return;
    }
  } catch (error) {
    showToast(t('server_error'), 'error');
    console.error(error);
  }
}

async function redirectAuthenticatedUsers() {
  const token = getToken();
  if (!token) {
    return;
  }
  const user = await getCurrentUser();
  if (user) {
    window.location.href = '/predict.html';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  redirectAuthenticatedUsers();

  const loginForm = document.getElementById('loginForm');
  const registerForm = document.getElementById('registerForm');

  if (loginForm) {
    loginForm.dataset.endpoint = '/api/login';
    loginForm.addEventListener('submit', handleFormSubmit);
  }

  if (registerForm) {
    registerForm.dataset.endpoint = '/api/register';
    registerForm.addEventListener('submit', handleFormSubmit);
  }
});
