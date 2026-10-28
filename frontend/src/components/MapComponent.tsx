import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, Rectangle, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import { PredictResponse, HistoricalFloodEvent, RiskCategory } from '../types';

// Fix Leaflet default icon issues in React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Custom pin marker icon
const customLocationIcon = new L.DivIcon({
  className: 'custom-div-icon',
  html: `
    <div style="background: #06b6d4; width: 18px; height: 18px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 15px #06b6d4;"></div>
  `,
  iconSize: [18, 18],
  iconAnchor: [9, 9]
});

const floodEventIcon = new L.DivIcon({
  className: 'flood-event-icon',
  html: `
    <div style="background: #f43f5e; width: 14px; height: 14px; border-radius: 50%; border: 2px solid #ffe4e6; box-shadow: 0 0 10px #f43f5e;"></div>
  `,
  iconSize: [14, 14],
  iconAnchor: [7, 7]
});

interface MapComponentProps {
  selectedLat: number;
  selectedLon: number;
  onLocationSelect: (lat: number, lon: number) => void;
  prediction: PredictResponse | null;
  historicalEvents: HistoricalFloodEvent[];
  showRiskOverlay: boolean;
  showHistoricalOverlay: boolean;
  showSatelliteCoverage: boolean;
}

// Controller to smoothly pan to newly selected coordinates
const MapRecenter: React.FC<{ lat: number; lon: number }> = ({ lat, lon }) => {
  const map = useMap();
  useEffect(() => {
    map.flyTo([lat, lon], Math.max(map.getZoom(), 7), { duration: 1.2 });
  }, [lat, lon, map]);
  return null;
};

// Map click listener
const MapEventsHandler: React.FC<{ onSelect: (lat: number, lon: number) => void }> = ({ onSelect }) => {
  useMapEvents({
    click(e) {
      onSelect(Number(e.latlng.lat.toFixed(4)), Number(e.latlng.lng.toFixed(4)));
    },
  });
  return null;
};

export const MapComponent: React.FC<MapComponentProps> = ({
  selectedLat,
  selectedLon,
  onLocationSelect,
  prediction,
  historicalEvents,
  showRiskOverlay,
  showHistoricalOverlay,
  showSatelliteCoverage
}) => {
  const getRiskColor = (cat: RiskCategory) => {
    switch (cat) {
      case 'low': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'high': return '#f97316';
      case 'very_high': return '#ef4444';
      default: return '#64748b';
    }
  };

  const riskColor = prediction ? getRiskColor(prediction.risk_category) : '#06b6d4';

  const [basemap, setBasemap] = React.useState<'street' | 'satellite' | 'topo'>('street');

  const tileConfigs = {
    street: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      attribution: 'Tiles &copy; Esri &mdash; Open Public Access'
    },
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: 'Tiles &copy; Esri Satellite &mdash; High-Resolution Aerial & Radar Context'
    },
    topo: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      attribution: 'Tiles &copy; Esri Topo &mdash; Topographic Wetness & Elevation'
    }
  };

  return (
    <div className="relative w-full h-full min-h-[500px] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950">
      {/* Basemap Switcher Controls */}
      <div className="absolute top-4 right-4 z-20 flex gap-1 bg-slate-900/90 backdrop-blur-md p-1 rounded-xl border border-slate-800 shadow-lg text-[11px] font-medium text-slate-300">
        <button
          type="button"
          onClick={() => setBasemap('street')}
          className={`px-2.5 py-1 rounded-lg transition-colors ${
            basemap === 'street' ? 'bg-cyan-600 text-white shadow-sm' : 'hover:bg-slate-800 text-slate-400'
          }`}
        >
          Street
        </button>
        <button
          type="button"
          onClick={() => setBasemap('satellite')}
          className={`px-2.5 py-1 rounded-lg transition-colors ${
            basemap === 'satellite' ? 'bg-cyan-600 text-white shadow-sm' : 'hover:bg-slate-800 text-slate-400'
          }`}
        >
          Satellite
        </button>
        <button
          type="button"
          onClick={() => setBasemap('topo')}
          className={`px-2.5 py-1 rounded-lg transition-colors ${
            basemap === 'topo' ? 'bg-cyan-600 text-white shadow-sm' : 'hover:bg-slate-800 text-slate-400'
          }`}
        >
          Topographic
        </button>
      </div>

      <MapContainer
        center={[selectedLat, selectedLon]}
        zoom={6}
        scrollWheelZoom={true}
        className="w-full h-full z-10"
        style={{ minHeight: '100%', height: '100%' }}
      >
        <MapRecenter lat={selectedLat} lon={selectedLon} />
        <MapEventsHandler onSelect={onLocationSelect} />

        {/* Free high-resolution map tiles (No API key or watermark required) */}
        <TileLayer
          key={basemap}
          attribution={tileConfigs[basemap].attribution}
          url={tileConfigs[basemap].url}
        />

        {/* Selected target marker */}
        <Marker position={[selectedLat, selectedLon]} icon={customLocationIcon}>
          <Popup className="custom-popup">
            <div className="p-1 font-sans text-xs">
              <strong className="text-cyan-600 block font-bold">Selected Coordinate</strong>
              <span>Lat: {selectedLat.toFixed(4)}° | Lon: {selectedLon.toFixed(4)}°</span>
              {prediction && (
                <div className="mt-1 pt-1 border-t border-slate-200">
                  <span className="capitalize font-semibold block" style={{ color: riskColor }}>
                    Risk: {prediction.risk_category}
                  </span>
                  <span>Probability: {((prediction.flood_probability || 0) * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          </Popup>
        </Marker>

        {/* Multimodal 48h Model Prediction Risk Overlay */}
        {showRiskOverlay && prediction && prediction.risk_category !== 'unavailable' && (
          <Circle
            center={[selectedLat, selectedLon]}
            radius={25000} // 25km radius footprint
            pathOptions={{
              color: riskColor,
              fillColor: riskColor,
              fillOpacity: 0.28,
              weight: 2,
              dashArray: '4, 6'
            }}
          >
            <Popup>
              <div className="text-xs">
                <span className="font-bold block uppercase" style={{ color: riskColor }}>
                  48-Hour Flood Risk Footprint
                </span>
                <span>Category: <strong>{prediction.risk_category.toUpperCase()}</strong></span><br />
                <span>Calibrated Probability: <strong>{((prediction.flood_probability || 0) * 100).toFixed(1)}%</strong></span>
              </div>
            </Popup>
          </Circle>
        )}

        {/* Sentinel-1 SAR Approximate Swath Footprint */}
        {showSatelliteCoverage && (
          <Rectangle
            bounds={[
              [selectedLat - 0.7, selectedLon - 1.2],
              [selectedLat + 0.7, selectedLon + 1.2]
            ]}
            pathOptions={{
              color: '#06b6d4',
              fillColor: '#06b6d4',
              fillOpacity: 0.08,
              weight: 1.5,
              dashArray: '2, 4'
            }}
          >
            <Popup>
              <div className="text-xs">
                <strong className="text-cyan-500 block">Copernicus Sentinel-1 SAR IW Footprint</strong>
                <span>Polarization: VV + VH | Resolution: 10m</span>
              </div>
            </Popup>
          </Rectangle>
        )}

        {/* Historical Flood Event Overlays */}
        {showHistoricalOverlay && historicalEvents.map((evt) => (
          <Marker
            key={evt.event_id}
            position={[evt.latitude, evt.longitude]}
            icon={floodEventIcon}
          >
            <Popup>
              <div className="text-xs max-w-xs">
                <strong className="text-rose-600 block font-bold">{evt.region_name} ({evt.country})</strong>
                <span className="text-slate-500 block">{evt.start_date} to {evt.end_date}</span>
                <span className="block mt-1">Severity Class: {evt.severity_class} ({evt.main_cause})</span>
                {evt.flooded_area_sq_km && <span>Flooded Area: {evt.flooded_area_sq_km.toLocaleString()} km²</span>}
                <div className="mt-1 pt-1 border-t text-[10px] text-slate-400">
                  Source: {evt.source_database}
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>

      {/* Floating Map Legend */}
      <div className="absolute bottom-4 right-4 z-20 bg-slate-900/90 backdrop-blur-md p-3 rounded-xl border border-slate-800 shadow-xl text-xs space-y-1.5 text-slate-200">
        <div className="font-semibold text-slate-300 text-[11px] uppercase tracking-wider mb-1 flex items-center justify-between">
          <span>Flood Risk Scale</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
          <span>Low (&lt; 25%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-amber-500"></span>
          <span>Moderate (25% - 55%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-orange-500"></span>
          <span>High (55% - 80%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-rose-500"></span>
          <span>Very High (&gt; 80%)</span>
        </div>
        <div className="pt-1.5 border-t border-slate-800 flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 border border-white"></span>
          <span className="text-[11px] text-slate-400">Recorded Flood Event</span>
        </div>
      </div>
    </div>
  );
};
