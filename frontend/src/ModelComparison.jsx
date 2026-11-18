import { useEffect, useState } from 'react';
import { request, apiUrl } from './api';

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    request('/api/model-comparison', { signal: controller.signal }).then(setData).catch((err) => {
      if (err.name !== 'AbortError') setError(err.message);
    });
    return () => controller.abort();
  }, []);
  return <section id="model-comparison">
    <div className="section-title"><div><h2>Model Comparison</h2><p>Measured results from 80 training trips and 10 student-test trips.</p></div></div>
    {error && <p className="feedback error" role="alert">Comparison evidence unavailable: {error}</p>}
    {!data && !error && <p className="feedback info">Loading experiment results…</p>}
    {data && <>
      <div className="panel performance-panel">
        <p className="metric-note">Selection uses trip-grouped cross-validation candidate Macro F1. Student-test results are comparison evidence. The tuned histogram boosting model is selected.</p>
        <table><thead><tr><th>Model</th><th>CV candidate F1</th><th>Test Top-1</th><th>Test candidate F1</th><th>Log loss</th></tr></thead>
          <tbody>{data.models.map((model) => <tr key={model.model} className={model.model.endsWith('__tuned') ? 'selected-model' : ''}>
            <td>{model.label}</td><td>{model.cv_candidate_macro_f1.toFixed(4)}</td>
            <td>{(model.student_test_top1_accuracy * 100).toFixed(2)}%</td>
            <td>{model.student_test_candidate_macro_f1.toFixed(4)}</td>
            <td>{model.student_test_candidate_log_loss.toFixed(4)}</td>
          </tr>)}</tbody></table>
      </div>
      <div className="comparison-graphs">{data.plots.map((plot) => <figure className="panel" key={plot.name}>
        <img src={apiUrl(`/api/model-comparison/plots/${plot.name}`)} alt={plot.title} loading="lazy" />
        <figcaption>{plot.title}</figcaption>
      </figure>)}</div>
    </>}
  </section>;
}
