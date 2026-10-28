import React, { useState, useEffect } from 'react';
import { MapComponent } from '../components/MapComponent';
import { fetchFloodHistory } from '../services/api';
import { HistoricalFloodEvent } from '../types';
import { Globe } from 'lucide-react';

export const GlobalMapPage: React.FC = () => {
  const [lat, setLat] = useState(20.0);
  const [lon, setLon] = useState(0.0);
  const [events, setEvents] = useState<HistoricalFloodEvent[]>([]);

  useEffect(() => {
    fetchFloodHistory(0.0, 0.0, 20000).then((res) => {
      setEvents(res.events || []);
    }).catch(console.error);
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-900 border border-slate-800 p-4 rounded-2xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400">
            <Globe className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">Global Flood Risk & Historical Event Atlas</h2>
            <p className="text-xs text-slate-400">
              Interactive multi-decadal record of inundation footprints and global observation swaths
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono text-slate-300">
          <span className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800">
            Current Cursor: {lat.toFixed(2)}°, {lon.toFixed(2)}°
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-300">
            {events.length} Major Recorded Events
          </span>
        </div>
      </div>

      <div className="h-[680px] w-full">
        <MapComponent
          selectedLat={lat}
          selectedLon={lon}
          onLocationSelect={(nLat, nLon) => {
            setLat(nLat);
            setLon(nLon);
          }}
          prediction={null}
          historicalEvents={events}
          showRiskOverlay={false}
          showHistoricalOverlay={true}
          showSatelliteCoverage={true}
        />
      </div>
    </div>
  );
};
