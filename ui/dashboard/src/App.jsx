import React, { useState, useEffect } from 'react';
import {
  Navigation,
  Activity,
  Cpu,
  Clock,
  Layers,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Server,
  Settings,
  ChevronRight,
  TrendingUp,
  Sliders,
  Compass,
  FileCode,
  Sparkles
} from 'lucide-react';
import MapComponent from './components/MapComponent';

// Default reserved trips from guidelines
const DEFAULT_TRIPS = [
  '00000000', '00000007', '00000009', '00000040', '00000052',
  '00000056', '00000065', '00000076', '00000077', '00000091'
];

export default function App() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [trips, setTrips] = useState(DEFAULT_TRIPS);
  const [selectedTrip, setSelectedTrip] = useState(DEFAULT_TRIPS[0]);
  const [loading, setLoading] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [latencyPing, setLatencyPing] = useState(null);

  // Results State
  const [evaluationData, setEvaluationData] = useState(null);
  const [error, setError] = useState(null);

  // Map Layer Controls
  const [showGps, setShowGps] = useState(true);
  const [showRoute, setShowRoute] = useState(true);
  const [showClassifier, setShowClassifier] = useState(false);

  // Trajectory Playback
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackIndex, setPlaybackIndex] = useState(-1);

  // Ping API on load
  useEffect(() => {
    checkApiConnection();
  }, [apiUrl]);

  const checkApiConnection = async () => {
    const start = performance.now();
    try {
      const res = await fetch(`${apiUrl}/health`, { method: 'GET' });
      if (res.ok) {
        setApiConnected(true);
        setLatencyPing(Math.round(performance.now() - start));
        fetchTrips();
      } else {
        setApiConnected(false);
      }
    } catch (e) {
      setApiConnected(false);
      setLatencyPing(null);
    }
  };

  const fetchTrips = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/trips`);
      if (res.ok) {
        const data = await res.json();
        if (data.trips && data.trips.length > 0) {
          setTrips(data.trips);
        }
      }
    } catch (e) {
      console.warn("Could not fetch remote trips list", e);
    }
  };

  const runEvaluation = async () => {
    if (!selectedTrip) return;
    setLoading(true);
    setError(null);
    setIsPlaying(false);
    setPlaybackIndex(-1);

    try {
      const res = await fetch(`${apiUrl}/api/predict/${selectedTrip}`, {
        method: 'POST',
      });
      if (!res.ok) {
        throw new Error(`API Error ${res.status}: ${await res.text()}`);
      }
      const data = await res.json();
      setEvaluationData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Playback Animation Loop
  useEffect(() => {
    let interval = null;
    if (isPlaying && evaluationData?.gps_display) {
      interval = setInterval(() => {
        setPlaybackIndex((prev) => {
          if (prev >= evaluationData.gps_display.length - 1) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 50);
    }
    return () => clearInterval(interval);
  }, [isPlaying, evaluationData]);

  const metrics = evaluationData?.metrics || {};

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      
      {/* Top Header Bar */}
      <header className="h-16 border-b border-slate-800/80 px-6 flex items-center justify-between glass-panel sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Navigation className="w-5 h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold tracking-tight text-white">PathSnap</h1>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Team 11 Production
              </span>
            </div>
            <p className="text-xs text-slate-400">Autonomous GPS-to-Road Map Matching & Viterbi Decoder</p>
          </div>
        </div>

        {/* Server Status & Config */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs font-mono bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <div className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-400">Laptop 1 API:</span>
            <span className={apiConnected ? 'text-emerald-400' : 'text-rose-400'}>
              {apiConnected ? `Connected (${latencyPing}ms)` : 'Disconnected'}
            </span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-900/60 rounded-lg p-1 border border-slate-800">
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="bg-transparent text-xs text-slate-300 px-2 py-1 focus:outline-none w-44 font-mono"
              placeholder="http://<IP>:8000"
            />
            <button
              onClick={checkApiConnection}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition"
              title="Ping Server"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-6 grid grid-cols-12 gap-6 max-w-[1920px] mx-auto w-full">
        
        {/* Left Column: Trip Selector & Telemetry (4 Cols) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-6">
          
          {/* Trip Selection Card */}
          <div className="glass-panel rounded-2xl p-5 flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Reserved Evaluation Trips</h2>
              </div>
              <span className="text-xs text-slate-400 font-mono">10 Reserved</span>
            </div>

            {/* Trip Grid Buttons */}
            <div className="grid grid-cols-2 gap-2">
              {trips.map((id) => (
                <button
                  key={id}
                  onClick={() => setSelectedTrip(id)}
                  className={`px-3 py-2.5 rounded-xl text-xs font-mono font-medium transition-all text-left flex items-center justify-between border ${
                    selectedTrip === id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 border-cyan-500/50 text-white shadow-lg shadow-cyan-500/10'
                      : 'bg-slate-900/50 border-slate-800/80 text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  }`}
                >
                  <span>Trip #{id}</span>
                  {selectedTrip === id && <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />}
                </button>
              ))}
            </div>

            {/* Run Button */}
            <button
              onClick={runEvaluation}
              disabled={loading}
              className={`w-full py-3.5 rounded-xl font-semibold text-sm transition-all flex items-center justify-center space-x-2 shadow-xl ${
                loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500 hover:opacity-95 text-slate-950 font-bold shadow-cyan-500/25 active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                  <span>Processing Neural Route Reconstruction...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute Map-Matching Pipeline</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Telemetry Metrics Panel */}
          <div className="glass-panel rounded-2xl p-5 flex flex-col space-y-4 flex-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Evaluation Telemetry</h2>
              </div>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Server-Side Verified
              </span>
            </div>

            {/* Metric Cards Grid */}
            <div className="grid grid-cols-2 gap-3">
              
              {/* Top-1 Accuracy */}
              <div className="glass-card rounded-xl p-3.5 flex flex-col justify-between">
                <div className="flex items-center justify-between text-slate-400 text-xs">
                  <span>Top-1 Accuracy</span>
                  <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
                </div>
                <div className="mt-2">
                  <div className="text-2xl font-bold font-mono text-white">
                    {metrics.top1_accuracy !== undefined && metrics.top1_accuracy !== null
                      ? `${(metrics.top1_accuracy * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">Reference Route Overlap</p>
                </div>
              </div>

              {/* Route Continuity */}
              <div className="glass-card rounded-xl p-3.5 flex flex-col justify-between">
                <div className="flex items-center justify-between text-slate-400 text-xs">
                  <span>Route Continuity</span>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                </div>
                <div className="mt-2">
                  <div className="text-2xl font-bold font-mono text-emerald-400">
                    {metrics.route_continuity !== undefined
                      ? `${(metrics.route_continuity * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">Graph Transition Validity</p>
                </div>
              </div>

              {/* Inference Latency */}
              <div className="glass-card rounded-xl p-3.5 flex flex-col justify-between">
                <div className="flex items-center justify-between text-slate-400 text-xs">
                  <span>Inference Latency</span>
                  <Clock className="w-3.5 h-3.5 text-amber-400" />
                </div>
                <div className="mt-2">
                  <div className="text-2xl font-bold font-mono text-white">
                    {metrics.inference_time_ms ? `${metrics.inference_time_ms}ms` : '—'}
                  </div>
                  <p className="text-[10px] text-emerald-400 font-medium mt-0.5">
                    {metrics.inference_time_ms && metrics.inference_time_ms < 150 ? '● Real-Time Ultra-Fast' : 'Batch Evaluation'}
                  </p>
                </div>
              </div>

              {/* Observation Volume */}
              <div className="glass-card rounded-xl p-3.5 flex flex-col justify-between">
                <div className="flex items-center justify-between text-slate-400 text-xs">
                  <span>Reconstructed Arcs</span>
                  <Cpu className="w-3.5 h-3.5 text-purple-400" />
                </div>
                <div className="mt-2">
                  <div className="text-2xl font-bold font-mono text-white">
                    {metrics.predicted_route_length || '—'}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {metrics.observations_count ? `from ${metrics.observations_count} GPS points` : 'Points Decoded'}
                  </p>
                </div>
              </div>
            </div>

            {/* Model Architecture Info */}
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-1.5 font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Active Model:</span>
                <span className="text-slate-200">LightGBM / Spatial Viterbi</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Coordinate System:</span>
                <span className="text-slate-200">Trip-Local Projected UTM</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Ground Truth Integrity:</span>
                <span className="text-emerald-400">Server-Side Protected</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Map & Trajectory Timeline (8 Cols) */}
        <div className="col-span-12 lg:col-span-8 flex flex-col space-y-4">
          
          {/* Map Controls Header */}
          <div className="glass-panel rounded-2xl p-3.5 flex items-center justify-between">
            <div className="flex items-center space-x-6">
              
              {/* Layer Toggles */}
              <div className="flex items-center space-x-2 text-xs">
                <Layers className="w-4 h-4 text-slate-400" />
                <span className="text-slate-400 font-medium">Layers:</span>
              </div>

              <label className="flex items-center space-x-2 text-xs cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showRoute}
                  onChange={(e) => setShowRoute(e.target.checked)}
                  className="rounded border-slate-700 text-emerald-500 focus:ring-0 bg-slate-800"
                />
                <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                  <span className="w-2.5 h-1 bg-emerald-400 rounded-full inline-block"></span>
                  Reconstructed Route
                </span>
              </label>

              <label className="flex items-center space-x-2 text-xs cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showGps}
                  onChange={(e) => setShowGps(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0 bg-slate-800"
                />
                <span className="text-cyan-400 font-semibold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block"></span>
                  Raw GPS Points
                </span>
              </label>

              <label className="flex items-center space-x-2 text-xs cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showClassifier}
                  onChange={(e) => setShowClassifier(e.target.checked)}
                  className="rounded border-slate-700 text-amber-500 focus:ring-0 bg-slate-800"
                />
                <span className="text-amber-400 font-semibold flex items-center gap-1.5">
                  <span className="w-2.5 h-1 border-t-2 border-dashed border-amber-400 inline-block"></span>
                  Raw Classifier (Before HMM)
                </span>
              </label>
            </div>

            {/* Current Trip Indicator */}
            <div className="text-xs font-mono text-slate-400">
              Trip Extent: <span className="text-white font-semibold">#{selectedTrip}</span>
            </div>
          </div>

          {/* Interactive Leaflet Map */}
          <div className="flex-1 min-h-[580px] w-full relative">
            <MapComponent
              gpsDisplay={evaluationData?.gps_display || []}
              routeCoords={evaluationData?.route_coords || []}
              classifierCoords={evaluationData?.classifier_coords || []}
              showGps={showGps}
              showRoute={showRoute}
              showClassifier={showClassifier}
              playbackIndex={playbackIndex}
            />

            {/* Overlay Watermark Badge */}
            <div className="absolute top-4 right-4 z-[400] glass-panel px-3 py-1.5 rounded-xl text-[11px] font-mono text-slate-300 flex items-center space-x-2 border border-slate-700/60 shadow-xl">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
              <span>CartoDB Dark Matter • Metric Proj</span>
            </div>
          </div>

          {/* Bottom Trajectory Scrubber / Playback Controls */}
          {evaluationData?.gps_display && evaluationData.gps_display.length > 0 && (
            <div className="glass-panel rounded-2xl p-4 flex items-center space-x-4">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-10 h-10 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center transition shrink-0"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
              </button>

              <div className="flex-1 flex flex-col space-y-1">
                <div className="flex justify-between text-xs font-mono text-slate-400">
                  <span>Trajectory Progress</span>
                  <span>
                    Observation {playbackIndex >= 0 ? playbackIndex : 0} / {evaluationData.gps_display.length}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={evaluationData.gps_display.length - 1}
                  value={playbackIndex >= 0 ? playbackIndex : 0}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setPlaybackIndex(parseInt(e.target.value));
                  }}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <button
                onClick={() => {
                  setIsPlaying(false);
                  setPlaybackIndex(0);
                }}
                className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition"
                title="Reset Scrubber"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
