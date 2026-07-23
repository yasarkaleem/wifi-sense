import ConnectionBadge from './components/ConnectionBadge.jsx';
import CountCard from './components/CountCard.jsx';
import HistoryChart from './components/HistoryChart.jsx';
import MotionGauge from './components/MotionGauge.jsx';
import ZoneHeatmap from './components/ZoneHeatmap.jsx';
import { useLiveData } from './useLiveData.js';

export default function App() {
  const { current, history, connectionStatus } = useLiveData();

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">wifi-sense</h1>
        <ConnectionBadge status={connectionStatus} />
      </header>

      <main className="app__grid">
        <CountCard count={current.count} confidence={current.confidence} presence={current.presence} />
        <MotionGauge value={current.motion_intensity} presence={current.presence} />
        <ZoneHeatmap zones={current.zones} timestamp={current.timestamp} />
        <HistoryChart history={history} />
      </main>
    </div>
  );
}
