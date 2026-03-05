import React, { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { Send, AlertTriangle } from "lucide-react";
import { Button } from "./ui/button";
import { useStore } from "../store/useStore";
import { PlaybackTimeline } from "./PlaybackTimeline";

export function MapCanvas() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapContent = useRef<maplibregl.Map | null>(null);
  const [disruptionText, setDisruptionText] = useState("");
  const { isRunning, simTime, setSimTime, timeBounds } = useStore();

  useEffect(() => {
    if (!mapContainer.current) return;
    mapContent.current = new maplibregl.Map({
      container: mapContainer.current,
      style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
      center: [8.6512, 49.8728],
      zoom: 13,
      pitch: 45,
      bearing: -17.6
    });
    return () => mapContent.current?.remove();
  }, []);

  // Simple interval to simulate playback when isRunning is true
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setSimTime(Math.min(simTime + 0.05, timeBounds[1]));
    }, 1000);
    return () => clearInterval(interval);
  }, [isRunning, simTime, timeBounds, setSimTime]);

  const handleInjectDisruption = () => {
    if(!disruptionText.trim() || !isRunning) return;
    console.log("Injecting GenAI Disruption:", disruptionText);
    setDisruptionText("");
  };

  return (
    <div className="w-full h-full relative">
      {/* Map Engine */}
      <div ref={mapContainer} className="w-full h-full absolute inset-0" />
      
      {/* Playback HUD (Top Center) */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-20">
        <PlaybackTimeline />
      </div>

      {/* GenAI Disruption Injector (Floating Bottom Center) */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 w-lg">
        <div className={`p-4 rounded-2xl border backdrop-blur-xl shadow-2xl transition-all duration-300 ${isRunning ? 'bg-zinc-950/80 border-zinc-700/50' : 'bg-zinc-950/40 border-zinc-800/30 opacity-70 pointer-events-none grayscale'}`}>
          <div className="flex items-center gap-2 mb-3 px-1">
             <AlertTriangle className={`w-4 h-4 ${isRunning ? 'text-amber-500' : 'text-zinc-500'}`} />
             <h2 className={`text-xs font-bold uppercase tracking-widest ${isRunning ? 'text-zinc-200' : 'text-zinc-500'}`}>GenAI Disruption Injector</h2>
          </div>
          <div className="relative flex gap-2">
            <input 
              disabled={!isRunning}
              value={disruptionText}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDisruptionText(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleInjectDisruption()}
              type="text"
              className="flex-1 bg-zinc-900/80 border border-zinc-700/80 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/50 disabled:bg-zinc-900 disabled:border-zinc-800 shadow-inner"
              placeholder="e.g. 'Protest on Rheinstraße blocking all lanes...'"
            />
            <Button 
              disabled={!isRunning || !disruptionText.trim()} 
              onClick={handleInjectDisruption}
              size="icon" 
              className={`shrink-0 rounded-xl h-10.5 w-10.5 transition-colors ${isRunning ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-600/20' : 'bg-zinc-800 text-zinc-600'}`}
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
 
