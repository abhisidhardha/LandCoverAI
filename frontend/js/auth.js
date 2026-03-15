async function handleFormSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const url = form.dataset.endpoint;

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = value.trim();
  });

  // Register form: ensure password match
  if (form.id === 'registerForm') {
    if (payload.password !== payload.confirm) {
      showToast('Passwords do not match.', 'error');
      return;
    }
    delete payload.confirm;
  }

  try {
    const res = await apiFetch(url, {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || 'Something went wrong.', 'error');
      return;
    }

    if (form.id === 'loginForm') {
      if (data.token) {
        setToken(data.token);
        showToast('Logged in successfully!', 'success');
        window.location.href = '/predict.html';
        return;
      }
    }

    if (form.id === 'registerForm') {
      showToast('Account created! Please log in.', 'success');
      window.location.href = '/login.html';
      return;
    }
  } catch (error) {
    showToast('Unable to reach server.', 'error');
    console.error(error);
  }
}

window.addEventListener('DOMContentLoaded', () => {
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
