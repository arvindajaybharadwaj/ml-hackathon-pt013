import { useEffect, useRef, useState } from 'react';
import { mapUrl, request } from './api';
import './App.css';

const partitionName = (partition) => partition === 'student_test' ? 'Student Test' :
  partition === 'train' ? 'Training' : 'Evaluator';
const percent = (value) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;

function App() {
  const [health, setHealth] = useState(null);
  const [trips, setTrips] = useState([]);
  const [tripId, setTripId] = useState('');
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [retry, setRetry] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const pending = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    async function connect() {
      try {
        const [backend, available] = await Promise.all([
          request('/health', { signal: controller.signal }),
          request('/api/trips', { signal: controller.signal }),
        ]);
        setHealth(backend);
        setTrips(available.trip_ids);
        setConnectionError('');
      } catch (err) {
        if (err.name !== 'AbortError') {
          setConnectionError('Cannot reach the backend. Start the API and retry the connection.');
        }
      }
    }
    connect();
    return () => controller.abort();
  }, [retry]);

  useEffect(() => () => pending.current?.abort(), []);

  async function handlePredict() {
    if (!tripId) {
      setError('Please select a trip before continuing.');
      return;
    }
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setError('');
    setPrediction(null);
    setEvaluation(null);
    setStatus('running');
    try {
      const result = await request(`/api/predict/${encodeURIComponent(tripId)}`, {
        method: 'POST', signal: controller.signal,
      });
      setPrediction(result);
      setStatus('complete');
      try {
        setEvaluation(await request(`/api/evaluate/${encodeURIComponent(tripId)}`, {
          signal: controller.signal,
        }));
      } catch (err) {
        if (err.name !== 'AbortError') setError(`Route is ready. Evaluation unavailable: ${err.message}`);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(`Prediction failed: ${err.message}`);
        setStatus('error');
      }
    }
  }

  const running = status === 'running';
  const connected = health && !connectionError;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">P</div>
          <div><h2>PathSnap</h2><span>INTELLIGENT MAP MATCHING</span></div>
        </div>
        <p className="nav-title">WORKSPACE</p>
        <a className="nav-link active" href="#dashboard">▦ &nbsp; Dashboard</a>
        <a className="nav-link" href="#route-map">⌖ &nbsp; Route Explorer</a>
        <a className="nav-link" href="#results">⌁ &nbsp; Predictions</a>
        <a className="nav-link" href="#performance">▤ &nbsp; Performance</a>
        <div className="sidebar-bottom">
          <span className={connected ? 'online-dot' : 'pending-dot'} />
          {connected ? 'Backend connected' : 'Connecting to backend'}
          <small>PT-013 · PathSnap</small>
        </div>
      </aside>

      <main className="main" id="dashboard">
        <header className="topbar">
          <span>Workspace / <strong>Dashboard</strong></span>
          <div className={`api-status ${connected ? 'connected' : ''}`} role="status">
            <span className={connected ? 'online-dot' : 'pending-dot'} />
            {connected ? (health.model_loaded ? 'Trained model connected' : 'Development fallback') : 'Backend offline'}
          </div>
        </header>
        <div className="content">
          <section className="heading">
            <div>
              <p className="eyebrow">DISTRIBUTED MAP-MATCHING PLATFORM</p>
              <h1>Route Reconstruction Dashboard</h1>
              <p>Transform noisy GPS observations into realistic vehicle routes.</p>
            </div>
            <div className="project-tag">PROJECT PT-013</div>
          </section>
          {connectionError && <div className="feedback error" role="alert">
            {connectionError} <button className="retry-button" onClick={() => setRetry(retry + 1)}>Retry connection</button>
          </div>}
          <section className="hero">
            <div>
              <span>WELCOME TO PATHSNAP</span>
              <h2>Understand the journey.<br />Reconstruct the route.</h2>
              <p>Select a real trip to compare model predictions with a route reconstructed using the road network.</p>
            </div>
            <div className="hero-art">⌁</div>
          </section>
          <section className="section-title">
            <div><h2>Trip Configuration</h2><p>{trips.length} trips available from the connected backend.</p></div>
            <span>STEP 01</span>
          </section>
          <section className="panel">
            <div className="field-grid">
              <label>DATASET
                <select value={health?.partition || ''} disabled>
                  <option value={health?.partition || ''}>{health ? `${partitionName(health.partition)} Dataset` : 'Waiting for backend'}</option>
                </select>
              </label>
              <label>SELECT TRIP
                <select value={tripId} disabled={!connected || running} onChange={(event) => {
                  pending.current?.abort();
                  setTripId(event.target.value); setStatus('idle');
                  setPrediction(null); setEvaluation(null); setError('');
                }}>
                  <option value="">Choose a trip</option>
                  {trips.map((id) => <option key={id} value={id}>Trip {id}</option>)}
                </select>
              </label>
            </div>
            <button className="predict-button" onClick={handlePredict} disabled={!connected || running}>
              {running ? 'Reconstructing route…' : 'Run Map Matching'} <span>→</span>
            </button>
            {running && <div className="feedback info" role="status">Scoring candidate roads and finding the most likely sequence. Large trips can take a little longer.</div>}
            {error && <div className="feedback error" role="alert">{error}</div>}
          </section>

          <section className="section-title" id="results">
            <div><h2>Prediction Overview</h2><p>{prediction ? `Trip ${prediction.trip_id} · ${prediction.predictor.replaceAll('_', ' ')}` : 'Select a trip to see its results.'}</p></div>
          </section>
          <section className="stats-grid">
            <div className="stat-card"><span>GPS OBSERVATIONS</span><h3>{prediction?.gps_observations.toLocaleString() || '—'}</h3><small>{tripId ? `Trip ${tripId}` : 'No trip selected'}</small></div>
            <div className="stat-card"><span>MATCHED OBSERVATIONS</span><h3>{prediction ? percent(prediction.matched_observations / prediction.gps_observations) : '—'}</h3><small>{prediction ? `${prediction.missing_observations} without candidates` : 'Awaiting prediction'}</small></div>
            <div className="stat-card"><span>PROCESSING TIME</span><h3>{prediction ? `${(prediction.processing_time_ms / 1000).toFixed(2)} s` : '—'}</h3><small>{status === 'complete' ? 'Route reconstruction complete' : running ? 'Processing' : 'Ready when you are'}</small></div>
          </section>

          <section className="section-title" id="route-map">
            <div><h2>Geographical Route Map</h2><p>Explore GPS observations, nearby roads and the predicted road segments.</p></div><span>MAP VIEW</span>
          </section>
          <section className="map-panel">
            {prediction ? <iframe className="route-frame" title={`Route map for trip ${prediction.trip_id}`} src={mapUrl(prediction.trip_id)} /> :
              <div className="map-placeholder"><div className="map-icon">⌖</div><h3>Your route will appear here</h3><p>Select a trip and run map matching to explore the reconstructed journey.</p></div>}
            <div className="legend"><span><i className="gps" />GPS observations</span><span><i className="road" />Road network</span><span><i className="route" />HMM matched roads</span></div>
          </section>

          <section className="section-title" id="performance"><div><h2>Route Performance</h2><p>Compare the classifier and sequence decoder. Reference data stays on the server.</p></div></section>
          <section className="panel performance-panel">
            {prediction ? <>
              <table><thead><tr><th>Metric</th><th>Classifier only</th><th>With HMM</th></tr></thead>
                <tbody>
                  <tr><td>Direct road continuity</td><td>{percent(prediction.classifier_only_route_continuity)}</td><td>{percent(prediction.route_continuity)}</td></tr>
                  <tr><td>Route sequence similarity</td><td>{percent(evaluation?.classifier_only_sequence_edit_similarity)}</td><td>{percent(evaluation?.sequence_edit_similarity)}</td></tr>
                  <tr><td>Route arc set F1</td><td>{percent(evaluation?.classifier_only_arc_set_f1)}</td><td>{percent(evaluation?.arc_set_f1)}</td></tr>
                </tbody></table>
              <p className="metric-note">Route metrics compare road sequences and road sets; they are different from GPS point accuracy.</p>
            </> : <p className="metric-note">Run a prediction to compare results.</p>}
          </section>
          <footer>PathSnap · Distributed Map-Matching & Route Reconstruction · PT-013</footer>
        </div>
      </main>
    </div>
  );
}
export default App;
