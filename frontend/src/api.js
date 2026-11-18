const base = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export async function request(path, options = {}) {
  const response = await fetch(`${base}${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch { /* Non-JSON proxy error. */ }
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return response.json();
}

export function mapUrl(tripId) {
  return `${base}/api/maps/${encodeURIComponent(tripId)}`;
}

export function apiUrl(path) { return `${base}${path}`; }
