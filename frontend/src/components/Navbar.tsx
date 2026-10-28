import React from 'react';
import { Waves, Activity, Cpu, ShieldAlert, Database, BarChart3, Info, Map as MapIcon } from 'lucide-react';
import { ModelStatus } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  backendOnline: boolean;
  modelStatus: ModelStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  backendOnline,
  modelStatus
}) => {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'map', label: 'Global Map', icon: MapIcon },
    { id: 'history', label: 'Historical Events', icon: Database },
    { id: 'performance', label: 'Model Performance', icon: Activity },
    { id: 'about', label: 'About & Methodology', icon: Info },
  ];

  return (
    <header className="bg-slate-900/90 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/20">
              <Waves className="w-6 h-6 animate-pulse" />
              <div className="absolute inset-0 rounded-xl border border-cyan-400/40 animate-ping opacity-25"></div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500 bg-clip-text text-transparent">
                  GlobalFlood AI
                </span>
                <span className="px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wider rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  48H FORECAST
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Multimodal SAR & Weather Deep Learning
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-blue-600/20 text-cyan-400 border border-cyan-500/30 shadow-sm'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Status Indicators */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Backend health */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 border border-slate-700">
              <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-slate-300 hidden sm:inline">{backendOnline ? 'API Online' : 'API Offline'}</span>
            </div>

            {/* Model status pill */}
            {modelStatus && (
              <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800/80 border border-slate-700">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span className="text-slate-300">
                  {modelStatus.is_model_trained
                    ? (modelStatus.is_smoke_test ? 'Model: Smoke v1' : 'Model: Trained')
                    : 'Model: Not Trained'}
                </span>
              </div>
            )}

            {/* Research Disclaimer Badge */}
            <div className="flex items-center gap-1 px-2 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-medium" title="Experimental Research Project - Not Official Emergency Warnings">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
              <span className="hidden xl:inline">Research Demo</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
