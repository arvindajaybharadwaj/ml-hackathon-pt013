
import { useState } from "react";
import "./App.css";

const evaluatorTrips = Array.from({ length: 10 }, (_, i) => ({
  value: `evaluator_${String(i + 1).padStart(2, "0")}`,
  label: `Evaluator Trip ${String(i + 1).padStart(2, "0")}`,
}));

function App() {
  const [dataset, setDataset] = useState("evaluator");
  const [tripId, setTripId] = useState("");
  const [predictionStatus, setPredictionStatus] = useState("idle");

  const trips = evaluatorTrips;

  const handlePredict = () => {
    if (!tripId) {
      setPredictionStatus("select");
      return;
    }

    setPredictionStatus("pending");
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">P</div>
          <div>
            <h2>PathSnap</h2>
            <span>INTELLIGENT MAP MATCHING</span>
          </div>
        </div>

        <p className="nav-title">WORKSPACE</p>
        <div className="nav-link active">▦ &nbsp; Dashboard</div>
        <div className="nav-link">⌖ &nbsp; Route Explorer</div>
        <div className="nav-link">⌁ &nbsp; Predictions</div>
        <div className="nav-link">▤ &nbsp; Performance</div>

        <div className="sidebar-bottom">
          <span className="online-dot"></span>
          Frontend operational
          <small>PT-013 · PathSnap</small>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <span>Workspace / <strong>Dashboard</strong></span>
          <div className="api-status">
            <span className="pending-dot"></span>
            Backend integration pending
          </div>
        </header>

        <div className="content">
          <section className="heading">
            <div>
              <p className="eyebrow">DISTRIBUTED MAP-MATCHING PLATFORM</p>
              <h1>Route Reconstruction Dashboard</h1>
              <p>
                Transform noisy GPS observations into realistic vehicle routes.
              </p>
            </div>
            <div className="project-tag">PROJECT PT-013</div>
          </section>

          <section className="hero">
            <div>
              <span>WELCOME TO PATHSNAP</span>
              <h2>Understand the journey.<br />Reconstruct the route.</h2>
              <p>
                Select a trip and prepare it for intelligent map matching
                and geographical route reconstruction.
              </p>
            </div>
            <div className="hero-art">⌁</div>
          </section>

          <section className="section-title">
            <div>
              <h2>Trip Configuration</h2>
              <p>Choose an evaluation dataset and trip.</p>
            </div>
            <span>STEP 01</span>
          </section>

          <section className="panel">
            <div className="field-grid">
              <label>
                DATASET
                <select
                  value={dataset}
                  onChange={(e) => {
                    setDataset(e.target.value);
                    setTripId("");
                    setPredictionStatus("idle");
                  }}
                >
                  <option value="evaluator">Evaluator Dataset</option>
                  <option value="student_test">Student Test Dataset</option>
                </select>
              </label>

              <label>
                SELECT TRIP
                <select
                  value={tripId}
                  onChange={(e) => {
                    setTripId(e.target.value);
                    setPredictionStatus("idle");
                  }}
                >
                  <option value="">Choose a trip</option>
                  {trips.map((trip) => (
                    <option key={trip.value} value={trip.value}>
                      {trip.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <button className="predict-button" onClick={handlePredict}>
              Run Map Matching <span>→</span>
            </button>

            {predictionStatus === "select" && (
              <div className="feedback error">
                Please select a trip before continuing.
              </div>
            )}

            {predictionStatus === "pending" && (
              <div className="feedback info">
                Trip selected successfully. Live prediction will start
                after the FastAPI backend is connected.
              </div>
            )}
          </section>

          <section className="section-title">
            <div>
              <h2>Prediction Overview</h2>
              <p>Results will populate after successful API integration.</p>
            </div>
          </section>

          <section className="stats-grid">
            <div className="stat-card">
              <span>SELECTED DATASET</span>
              <h3>{dataset === "evaluator" ? "Evaluator" : "Student Test"}</h3>
              <small>Current partition</small>
            </div>
            <div className="stat-card">
              <span>SELECTED TRIP</span>
              <h3>{tripId ? tripId.replace("_", " ").toUpperCase() : "—"}</h3>
              <small>Trip identifier</small>
            </div>
            <div className="stat-card">
              <span>PREDICTION STATUS</span>
              <h3>
                {predictionStatus === "pending" ? "Awaiting API" : "Not started"}
              </h3>
              <small>No prediction generated yet</small>
            </div>
          </section>

          <section className="section-title">
            <div>
              <h2>Geographical Route Map</h2>
              <p>GPS observations, road network and reconstructed route.</p>
            </div>
            <span>MAP VIEW</span>
          </section>

          <section className="map-panel">
            <div className="map-placeholder">
              <div className="map-icon">⌖</div>
              <h3>Route visualisation</h3>
              <p>
                The interactive Folium map will appear here after
                map and backend integration.
              </p>
              <span className="map-badge">MAP MODULE READY FOR INTEGRATION</span>
            </div>
            <div className="legend">
              <span><i className="gps"></i> GPS observations</span>
              <span><i className="road"></i> Road network</span>
              <span><i className="route"></i> Predicted route</span>
            </div>
          </section>

          <footer>
            PathSnap · Distributed Map-Matching & Route Reconstruction · PT-013
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
