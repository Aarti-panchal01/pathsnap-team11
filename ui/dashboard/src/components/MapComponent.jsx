import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

export default function MapComponent({
  gpsDisplay = [],
  routeCoords = [],
  classifierCoords = [],
  showGps = true,
  showRoute = true,
  showClassifier = false,
  playbackIndex = -1,
  mapCenter = [54.73, 55.97],
  zoom = 13,
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const layersRef = useRef({
    gpsLayer: null,
    routeLayer: null,
    classifierLayer: null,
    vehicleMarker: null,
    startMarker: null,
    endMarker: null,
  });

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: mapCenter,
        zoom: zoom,
        zoomControl: false,
        attributionControl: false,
      });

      // High-performance clean dark tiles (No API key watermark required)
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 16,
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      }).addTo(map);

      // Add Zoom Control to bottom-right
      L.control.zoom({ position: 'bottomright' }).addTo(map);

      mapInstanceRef.current = map;
    }

    return () => {
      // Cleanup on unmount
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update Layers & Geometry when data changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const layers = layersRef.current;

    // Clear old layers
    if (layers.gpsLayer) map.removeLayer(layers.gpsLayer);
    if (layers.routeLayer) map.removeLayer(layers.routeLayer);
    if (layers.classifierLayer) map.removeLayer(layers.classifierLayer);
    if (layers.vehicleMarker) map.removeLayer(layers.vehicleMarker);
    if (layers.startMarker) map.removeLayer(layers.startMarker);
    if (layers.endMarker) map.removeLayer(layers.endMarker);

    let bounds = null;

    // 1. Render Reconstructed Route (Emerald Green Glow)
    if (showRoute && routeCoords && routeCoords.length > 0) {
      // Base glowing line
      const glowPolyline = L.polyline(routeCoords, {
        color: '#10b981',
        weight: 8,
        opacity: 0.35,
        lineCap: 'round',
        lineJoin: 'round',
      });

      // Core crisp line
      const corePolyline = L.polyline(routeCoords, {
        color: '#34d399',
        weight: 4,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round',
      });

      const routeGroup = L.featureGroup([glowPolyline, corePolyline]);
      routeGroup.addTo(map);
      layers.routeLayer = routeGroup;
      bounds = corePolyline.getBounds();

      // Start & End Markers
      if (routeCoords.length >= 2) {
        const startPt = routeCoords[0];
        const endPt = routeCoords[routeCoords.length - 1];

        const startIcon = L.divIcon({
          className: 'custom-pin',
          html: `<div class="flex items-center justify-center w-7 h-7 rounded-full bg-emerald-500/20 border-2 border-emerald-400 text-white font-bold text-xs shadow-lg shadow-emerald-500/40">S</div>`,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        });

        const endIcon = L.divIcon({
          className: 'custom-pin',
          html: `<div class="flex items-center justify-center w-7 h-7 rounded-full bg-rose-500/20 border-2 border-rose-400 text-white font-bold text-xs shadow-lg shadow-rose-500/40">F</div>`,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        });

        layers.startMarker = L.marker(startPt, { icon: startIcon }).addTo(map).bindPopup('<b>Route Origin</b>');
        layers.endMarker = L.marker(endPt, { icon: endIcon }).addTo(map).bindPopup('<b>Route Destination</b>');
      }
    }

    // 2. Render Classifier-Only Route (Comparison - Amber / Orange)
    if (showClassifier && classifierCoords && classifierCoords.length > 0) {
      const classPolyline = L.polyline(classifierCoords, {
        color: '#f59e0b',
        weight: 3,
        opacity: 0.7,
        dashArray: '6, 6',
      });
      classPolyline.addTo(map);
      layers.classifierLayer = classPolyline;
      if (!bounds) bounds = classPolyline.getBounds();
    }

    // 3. Render Raw GPS Observations (Cyan / Neon Blue Points)
    if (showGps && gpsDisplay && gpsDisplay.length > 0) {
      const gpsMarkers = gpsDisplay.map((pt) => {
        return L.circleMarker([pt.lat, pt.lon], {
          radius: 3.5,
          color: '#38bdf8',
          fillColor: '#0284c7',
          fillOpacity: 0.8,
          weight: 1.5,
        }).bindPopup(`<b>Observation #${pt.step}</b><br/>Lat: ${pt.lat.toFixed(6)}<br/>Lon: ${pt.lon.toFixed(6)}`);
      });

      const gpsGroup = L.featureGroup(gpsMarkers);
      gpsGroup.addTo(map);
      layers.gpsLayer = gpsGroup;

      if (!bounds) bounds = gpsGroup.getBounds();
    }

    // 4. Vehicle Playback Marker
    if (playbackIndex >= 0 && gpsDisplay && playbackIndex < gpsDisplay.length) {
      const currPt = gpsDisplay[playbackIndex];
      const vehicleIcon = L.divIcon({
        className: 'vehicle-marker',
        html: `
          <div class="relative flex items-center justify-center">
            <div class="absolute w-6 h-6 rounded-full bg-cyan-400/30 animate-ping"></div>
            <div class="w-4 h-4 rounded-full bg-cyan-400 border-2 border-white shadow-md shadow-cyan-500"></div>
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      layers.vehicleMarker = L.marker([currPt.lat, currPt.lon], { icon: vehicleIcon }).addTo(map);
    }

    // Auto-fit to trajectory bounds
    if (bounds && bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 16 });
    }
  }, [gpsDisplay, routeCoords, classifierCoords, showGps, showRoute, showClassifier, playbackIndex]);

  return (
    <div className="relative w-full h-full min-h-[550px] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
      <div ref={mapContainerRef} className="w-full h-full min-h-[550px]" />
    </div>
  );
}
