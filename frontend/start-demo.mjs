// Run on the UI laptop: node frontend/start-demo.mjs --backend http://192.168.1.10:8000
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const args = process.argv.slice(2);
const value = (name, fallback) => args.includes(name) ? args[args.indexOf(name)+1] : fallback;
const backend = new URL(value('--backend', 'http://127.0.0.1:8000'));
if (!['http:', 'https:'].includes(backend.protocol) || backend.username || backend.password || backend.search || backend.pathname !== '/') {
  throw new Error('Use a backend origin such as http://192.168.1.10:8000');
}
const port = value('--port', '5173');
if (!/^\d+$/.test(port) || +port < 1 || +port > 65535) throw new Error('Invalid frontend port');
try {
  const [health, trips] = await Promise.all(['/health', '/api/trips'].map(async (path) => {
    const response = await fetch(new URL(path, backend), { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }));
  if (!health.model_loaded || !health.dataset_available) throw new Error('Backend has no trained model or dataset');
  console.log(`Connected to ${backend.origin} | ${health.predictor} | ${trips.trip_ids.length} ${trips.partition} trips`);
} catch (err) {
  console.error(`Cannot use backend: ${err.message}. Check its LAN IP, API process and firewall.`);
  process.exit(1);
}
if (!args.includes('--check')) {
  const child = spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm',
    ['run','dev','--','--host','0.0.0.0','--port',port,'--strictPort'],
    { cwd: dirname(fileURLToPath(import.meta.url)), stdio: 'inherit',
      env: { ...process.env, VITE_BACKEND_TARGET: backend.origin } });
  for (const signal of ['SIGINT','SIGTERM']) process.on(signal, () => child.kill(signal));
  child.on('exit', (code) => process.exit(code ?? 0));
}
