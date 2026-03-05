import { Play, Pause, FastForward } from "lucide-react";
import { useStore } from "../store/useStore";
import { useState } from "react";

export function PlaybackTimeline() {
  const { isRunning, setIsRunning, isReadyToSimulate } = useStore();
  const [speed, setSpeed] = useState(1);

  const ready = isReadyToSimulate();

  return (
    <div className="flex flex-col items-center gap-2 pointer-events-auto w-full max-w-lg mx-auto">
      <div className="bg-zinc-950/90 border border-zinc-800/80 backdrop-blur-xl rounded-full px-5 py-2.5 flex items-center justify-between gap-6 shadow-2xl relative">
        <div className="flex items-center gap-4">
          <button 
            disabled={!ready}
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center justify-center p-2 rounded-full transition-all group ${!ready ? 'bg-zinc-800/50 cursor-not-allowed opacity-50 text-zinc-500' : isRunning ? 'bg-red-500/10 text-red-500 hover:bg-red-500/20' : 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'}`}
          >
            {isRunning ? (
              <Pause className="w-5 h-5 fill-current" />
            ) : (
              <Play className="w-5 h-5 fill-current ml-0.5" />
            )}
          </button>
          
          <button 
             disabled={!ready}
             onClick={() => setSpeed(s => s === 1 ? 2 : s === 2 ? 5 : 1)}
             className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold transition-colors ${!ready ? 'text-zinc-600 bg-zinc-900/50 cursor-not-allowed' : 'text-blue-400 bg-blue-500/10 hover:bg-blue-500/20'}`}>
            <FastForward className="w-3.5 h-3.5" />
            {speed}x
          </button>
        </div>

        <div className="w-px h-6 bg-zinc-800"></div>

        <div className="flex items-center gap-3">
           <div className="font-mono text-sm tracking-wider font-bold text-white">
             08:30 <span className="text-zinc-500 text-xs">AM</span>
           </div>
        </div>

        {/* Floating Tooltip if disabled */}
        {!ready && (
          <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 whitespace-nowrap bg-zinc-900 border border-zinc-700/50 px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-400 shadow-xl opacity-0 hover:opacity-100 transition-opacity flex flex-col items-center pointer-events-none group-hover:opacity-100">
             <div className="w-2 h-2 bg-zinc-900 border-l border-t border-zinc-700/50 absolute -top-1.5 rotate-45"></div>
             Upload Orders & Vehicles to Start
          </div>
        )}
      </div>

      <div className={`w-full relative px-2 ${!ready ? 'opacity-30 pointer-events-none cursor-not-allowed grayscale' : ''}`}>
        <input 
           disabled={!ready}
           type="range" 
           min="0" 
           max="100" 
           defaultValue="35"
           className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-400 [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(52,211,153,0.5)]"
        />
      </div>
    </div>
  );
}
