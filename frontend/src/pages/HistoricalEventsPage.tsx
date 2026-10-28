import React, { useState, useEffect } from 'react';
import { Database, Search, Filter, ExternalLink, Calendar, MapPin, Users } from 'lucide-react';
import { fetchFloodHistory } from '../services/api';
import { HistoricalFloodEvent } from '../types';

export const HistoricalEventsPage: React.FC = () => {
  const [events, setEvents] = useState<HistoricalFloodEvent[]>([]);
  const [search, setSearch] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');

  useEffect(() => {
    fetchFloodHistory(0, 0, 20000).then((res) => {
      setEvents(res.events || []);
    }).catch(console.error);
  }, []);

  const filteredEvents = events.filter((e) => {
    const matchesSearch =
      e.region_name.toLowerCase().includes(search.toLowerCase()) ||
      e.country.toLowerCase().includes(search.toLowerCase()) ||
      e.main_cause.toLowerCase().includes(search.toLowerCase());
    const matchesSeverity = selectedSeverity === 'all' || e.severity_class.toString() === selectedSeverity;
    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="space-y-6">
      {/* Title & Stats */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Database className="w-5 h-5 text-cyan-400" />
            <h2 className="text-xl font-bold text-white tracking-tight">Global Flood Event Archive</h2>
          </div>
          <p className="text-xs text-slate-400 max-w-2xl">
            Curated historical inundations compiled from the Global Flood Database (MODIS Events V1)
            and the Dartmouth Flood Observatory (DFO) Global Active Archive.
          </p>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="bg-slate-950 border border-slate-800 px-3 py-2 rounded-xl text-center">
            <span className="text-slate-400 block text-[10px] uppercase">Cataloged Events</span>
            <span className="text-lg font-bold text-cyan-400">{events.length}</span>
          </div>
          <div className="bg-slate-950 border border-slate-800 px-3 py-2 rounded-xl text-center">
            <span className="text-slate-400 block text-[10px] uppercase">Temporal Span</span>
            <span className="text-lg font-bold text-blue-400">2015 – 2024</span>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter by country, river basin, or hydrological cause..."
            className="w-full bg-slate-950 border border-slate-700 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Severities</option>
            <option value="1">Class 1: Large Flood</option>
            <option value="2">Class 2: Very Large</option>
            <option value="3">Class 3: Extreme Disaster</option>
          </select>
        </div>
      </div>

      {/* Event Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredEvents.map((evt) => (
          <div
            key={evt.event_id}
            className="bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all rounded-2xl p-5 shadow-lg flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-950 text-cyan-400 border border-slate-800">
                  {evt.event_id}
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                    evt.severity_class === 3
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  }`}
                >
                  Severity {evt.severity_class}
                </span>
              </div>

              <h3 className="text-base font-bold text-white mb-1">
                {evt.region_name}
              </h3>
              <p className="text-xs text-cyan-300 font-medium mb-3 flex items-center gap-1">
                <MapPin className="w-3.5 h-3.5" /> {evt.country}
              </p>

              <div className="space-y-2 text-xs text-slate-300 bg-slate-950 p-3 rounded-xl border border-slate-850 mb-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 flex items-center gap-1.5">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" /> Duration:
                  </span>
                  <span className="font-mono text-slate-200">
                    {evt.start_date} ({evt.duration_days} days)
                  </span>
                </div>
                {evt.flooded_area_sq_km && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Inundated Area:</span>
                    <span className="font-mono text-slate-200">
                      {evt.flooded_area_sq_km.toLocaleString()} km²
                    </span>
                  </div>
                )}
                {evt.displaced && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-slate-400" /> Displaced:
                    </span>
                    <span className="font-mono text-rose-400 font-semibold">
                      {evt.displaced.toLocaleString()} people
                    </span>
                  </div>
                )}
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed mb-4">
                <strong>Meteorological Trigger:</strong> {evt.main_cause}
              </p>
            </div>

            <div className="pt-3 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-500">
              <span className="truncate max-w-[200px]">{evt.source_database}</span>
              <span className="text-cyan-400 flex items-center gap-1 hover:underline cursor-pointer">
                Catalog Info <ExternalLink className="w-3 h-3" />
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
