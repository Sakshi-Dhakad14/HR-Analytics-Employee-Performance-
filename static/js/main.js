const charts = {};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawChart(id, config) {
  Chart.defaults.font.family = "Inter, 'Segoe UI', sans-serif";
  Chart.defaults.color = cssVar('--muted');
  Chart.defaults.borderColor = cssVar('--border');
  config.options = config.options || {};
  config.options.responsive = true;
  config.options.maintainAspectRatio = false;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

function filterQS() {
  const p = new URLSearchParams();
  document.querySelectorAll('.filter').forEach(s => {
    if (s.value && s.value !== 'All') p.set(s.name, s.value);
  });
  return p.toString();
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (r.redirected) { window.location.href = '/login'; return; }
  return r.json();
}

function money(n) { return '$' + Number(n).toLocaleString(); }
function badge(level) { return `<span class="badge ${level}">${level}</span>`; }
function riskColor(level) {
  return level === 'High' ? cssVar('--high') : level === 'Medium' ? cssVar('--med') : cssVar('--low');
}

// ---- Icons ----
function refreshIcons() { if (window.lucide) lucide.createIcons(); }
refreshIcons();

// ---- Mobile sidebar ----
const menuBtn = document.getElementById('menuBtn');
if (menuBtn) {
  menuBtn.addEventListener('click', () =>
    document.getElementById('sidebar').classList.toggle('open'));
}

// ---- Dark mode ----
const toggleBtn = document.getElementById('themeToggle');

function updateToggleLabel() {
  if (!toggleBtn) return;
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  toggleBtn.querySelector('span').textContent = dark ? 'Light mode' : 'Dark mode';
  toggleBtn.querySelector('i').setAttribute('data-lucide', dark ? 'sun' : 'moon');
  refreshIcons();
}

if (toggleBtn) {
  updateToggleLabel();
  toggleBtn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateToggleLabel();
    window.dispatchEvent(new Event('themechange'));
  });
}