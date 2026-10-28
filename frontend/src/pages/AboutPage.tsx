import React from 'react';
import {
  Waves,
  Cpu,
  Database,
  ShieldAlert,
  Satellite,
  Compass,
  CloudRain,
  ExternalLink,
  BookOpen
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12 text-slate-200">
      {/* Title */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl -z-0"></div>
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-3">
            <span className="p-2.5 rounded-xl bg-cyan-950 text-cyan-400 border border-cyan-800">
              <Waves className="w-6 h-6" />
            </span>
            <span className="px-3 py-1 text-xs font-mono font-bold uppercase rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
              Research Project Specification
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            GlobalFlood AI: Multimodal Deep Learning for 48-Hour Global Flood Forecasting
          </h1>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed max-w-3xl">
            An advanced college-level geospatial machine learning system fusing Copernicus Sentinel-1
            Synthetic Aperture Radar (SAR), numerical weather time-series, and high-resolution digital
            elevation models into a unified risk probability engine.
          </p>
        </div>
      </div>

      {/* Abstract and Problem Statement */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" /> Research Abstract
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Floods cause tens of billions of dollars in damage annually and displace millions worldwide.
            Traditional hydrodynamic models rely on dense river-gauge networks and computational fluid dynamics,
            rendering them inapplicable across gauge-sparse developing nations. GlobalFlood AI bridges this gap
            by combining global satellite radar telemetry, open meteorological reforecasts, and global topography.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" /> Problem Statement & Limitations
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Single-modality flood models fail under complex triggers: rainfall forecasts suffer from
            spatial displacement; optical imagery cannot penetrate storm clouds; elevation alone misses
            soil saturation. By synthesizing cloud-penetrating SAR radar backscatter with recurrent temporal
            weather sequences, our multimodal architecture models antecedent saturation and future precipitation accumulation.
          </p>
        </div>
      </div>

      {/* Multimodal Architecture Breakdown */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" /> Tri-Branch Multimodal Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 font-bold">
              <Satellite className="w-4 h-4" /> Branch 1: Satellite SAR CNN
            </div>
            <p className="text-slate-400 leading-relaxed">
              <strong>Input:</strong> 128×128 pixel dual-polarization (VV/VH) Sentinel-1 C-band SAR patch.
              Extracts 128-dimensional spatial radar backscatter embedding via 4 convolutional blocks + Global Average Pooling.
            </p>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center gap-2 text-indigo-400 font-bold">
              <CloudRain className="w-4 h-4" /> Branch 2: Weather GRU
            </div>
            <p className="text-slate-400 leading-relaxed">
              <strong>Input:</strong> 168h historical observations + 48h forecast sequence across 6 variables (precip, temp, humidity, pressure, wind, soil).
              Extracts 64-dimensional temporal dynamics with missingness masks.
            </p>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center gap-2 text-amber-400 font-bold">
              <Compass className="w-4 h-4" /> Branch 3: Geographic MLP
            </div>
            <p className="text-slate-400 leading-relaxed">
              <strong>Input:</strong> 9-dimensional vector: elevation, slope, terrain roughness index (TRI), topographic wetness index (TWI), river distance, and spherical sin/cos coordinates.
              Projects to 32 dimensions.
            </p>
          </div>
        </div>

        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-cyan-300">
          Fusion Layer: [SAR: 128d] + [Weather: 64d] + [Geo: 32d] + [Modality Flags: 3d] = 227-d
          &rarr; Dense(128, ReLU) &rarr; Dropout(0.3) &rarr; Dense(64, ReLU) &rarr; Logit &rarr; Platt Calibrated Probability
        </div>
      </div>

      {/* Data Sources and Attribution */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Database className="w-4 h-4 text-emerald-400" /> Verified Scientific Data Sources
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <strong className="text-white block">Open-Meteo API</strong>
              <span className="text-slate-400">Global Numerical Forecasts & Reanalysis Archives</span>
            </div>
            <a href="https://open-meteo.com" target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline">
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <strong className="text-white block">Copernicus Data Space (CDSE)</strong>
              <span className="text-slate-400">Sentinel-1 C-SAR IW GRD 10m Resolution</span>
            </div>
            <a href="https://dataspace.copernicus.eu" target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline">
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <strong className="text-white block">Global Flood Database (GFD)</strong>
              <span className="text-slate-400">Google Earth Engine MODIS Events Catalog V1</span>
            </div>
            <a href="https://global-flood-database.cloudtostreet.ai" target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline">
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex justify-between items-center">
            <div>
              <strong className="text-white block">Copernicus DEM & Open-Elevation</strong>
              <span className="text-slate-400">90m World Topographic Elevation Grid</span>
            </div>
            <a href="https://open-elevation.com" target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline">
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>

      {/* Research Disclaimer & Ethics */}
      <div className="bg-amber-950/20 border border-amber-500/30 rounded-2xl p-6 shadow-xl text-xs space-y-2">
        <h3 className="font-bold text-amber-300 uppercase tracking-wider flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-400" /> Academic & Operational Disclaimer
        </h3>
        <p className="text-amber-200/90 leading-relaxed">
          GlobalFlood AI is an experimental academic prototype developed for research evaluation in multimodal machine learning.
          Predictions generated by this system are probabilistic statistical estimates and <strong>DO NOT</strong> represent
          official disaster warnings, flood advisories, or evacuation orders. Real-time emergency response must strictly rely
          on authorized national weather services and local disaster management agencies.
        </p>
      </div>
    </div>
  );
};
