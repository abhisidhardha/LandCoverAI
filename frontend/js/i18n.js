// ── i18n Translation Engine ──────────────────────────────────
const SUPPORTED_LANGS = {
  en: { label: 'English', flag: '🇬🇧' },
  te: { label: 'తెలుగు', flag: '🇮🇳' },
  hi: { label: 'हिन्दी', flag: '🇮🇳' },
  ta: { label: 'தமிழ்', flag: '🇮🇳' },
  bn: { label: 'বাংলা', flag: '🇮🇳' },
  ml: { label: 'മലയാളം', flag: '🇮🇳' },
};

const LANG_STORAGE_KEY = 'landcoverai_lang';
let currentLang = localStorage.getItem(LANG_STORAGE_KEY) || 'en';

// Translation dictionaries loaded from separate files via i18n_translations.js
// window._i18nData is set by i18n_translations.js
function getTranslations() {
  return window._i18nData || {};
}

function t(key) {
  const data = getTranslations();
  const langData = data[currentLang] || data['en'] || {};
  return langData[key] || (data['en'] || {})[key] || key;
}

function getCurrentLang() {
  return currentLang;
}

function setLang(lang) {
  if (!SUPPORTED_LANGS[lang]) return;
  currentLang = lang;
  localStorage.setItem(LANG_STORAGE_KEY, lang);
  applyTranslations();
  // Update switcher display
  const sel = document.getElementById('langSwitcher');
  if (sel) sel.value = lang;
}

function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const val = t(key);
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      // skip
    } else {
      el.textContent = val;
    }
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    el.innerHTML = t(el.getAttribute('data-i18n-html'));
  });
  // Update page title if data attribute exists on html
  const titleEl = document.querySelector('title[data-i18n]');
  if (titleEl) titleEl.textContent = t(titleEl.getAttribute('data-i18n'));
}

function mountLangSwitcher() {
  const nav = document.querySelector('.nav');
  if (!nav || document.getElementById('langSwitcher')) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'lang-switcher-wrap';

  const select = document.createElement('select');
  select.id = 'langSwitcher';
  select.className = 'lang-switcher';
  select.setAttribute('aria-label', 'Select language');

  for (const [code, info] of Object.entries(SUPPORTED_LANGS)) {
    const opt = document.createElement('option');
    opt.value = code;
    opt.textContent = `${info.flag} ${info.label}`;
    if (code === currentLang) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener('change', () => setLang(select.value));
  wrapper.appendChild(select);
  nav.appendChild(wrapper);
}

window.addEventListener('DOMContentLoaded', () => {
  mountLangSwitcher();
  applyTranslations();
});
