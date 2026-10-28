import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { GlobalMapPage } from './pages/GlobalMapPage';
import { HistoricalEventsPage } from './pages/HistoricalEventsPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { AboutPage } from './pages/AboutPage';
import { checkBackendHealth, fetchModelStatus } from './services/api';
import { ModelStatus } from './types';

import { ErrorBoundary } from './components/ErrorBoundary';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendOnline, setBackendOnline] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    // Initial health and model check
    const pollHealth = async () => {
      try {
        await checkBackendHealth();
        setBackendOnline(true);
        const st = await fetchModelStatus();
        setModelStatus(st);
      } catch (err) {
        setBackendOnline(false);
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendOnline={backendOnline}
        modelStatus={modelStatus}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ErrorBoundary fallbackTitle="Application View Error">
          {activeTab === 'dashboard' && <DashboardPage />}
          {activeTab === 'map' && <GlobalMapPage />}
          {activeTab === 'history' && <HistoricalEventsPage />}
          {activeTab === 'performance' && <ModelPerformancePage />}
          {activeTab === 'about' && <AboutPage />}
        </ErrorBoundary>
      </main>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-slate-900 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-400">GlobalFlood AI</span>
            <span>&bull;</span>
            <span>Multimodal SAR & Weather Deep Learning Forecasting</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span className="text-[11px] bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-md font-mono">
              WGS84 Spherical CRS
            </span>
            <span className="text-[11px] bg-slate-900 border border-slate-800 px-2.5 py-1 rounded-md font-mono">
              PyTorch 2.x
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
