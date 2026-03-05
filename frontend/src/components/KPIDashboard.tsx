import React from "react";
import { useStore } from "../store/useStore";
import { Activity, Clock, Zap, Target } from "lucide-react";

export function KPIDashboard() {
  const { baselineStrategy, setBaselineStrategy, isRunning, activeMetrics } = useStore();

  const metrics = activeMetrics || {
    distance: { baseline: "-", current: "-" },
    utilization: { baseline: "-", current: "-" },
    sla: { baseline: "-", current: "-" }
  };

  const percChangeDist = activeMetrics 
    ? Math.round(((metrics.distance.current - metrics.distance.baseline) / metrics.distance.baseline) * 100) 
    : 0;

  return (
    <div className="w-75 pointer-events-auto bg-zinc-950/90 border border-zinc-800/80 backdrop-blur-xl p-5 shadow-2xl rounded-2xl flex flex-col gap-4 relative overflow-hidden">
      
      {!activeMetrics && (
        <div className="absolute inset-0 z-10 bg-zinc-950/60 backdrop-blur-[2px] flex items-center justify-center p-6 text-center">
          <p className="text-zinc-400 text-xs font-medium uppercase tracking-widest leading-relaxed">
            Run Simulation to generate metrics
          </p>
        </div>
      )}

      <div className="flex justify-between items-center mb-1">
        <h3 className="text-white font-bold text-sm tracking-wide flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          Live Telemetry
        </h3>
        <span className="flex h-2 w-2 relative">
          {isRunning && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>}
          <span className={`relative inline-flex rounded-full h-2 w-2 ${isRunning ? 'bg-emerald-500' : 'bg-zinc-600'}`}></span>
        </span>
      </div>

      <div className="bg-zinc-900/50 p-1.5 rounded-lg border border-zinc-800/50">
        <select 
          value={baselineStrategy}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setBaselineStrategy(e.target.value)}
          className="bg-transparent text-zinc-400 font-medium text-xs w-full outline-none px-1 py-1 cursor-pointer"
        >
          <option>Compare vs Legacy Baseline</option>
          <option>Compare vs Optimized Target</option>
        </select>
      </div>

      <div className="space-y-4">
        <div className="space-y-1">
          <div className="flex justify-between items-center text-xs text-zinc-500 font-medium">
            <span className="flex items-center gap-1.5"><Target className="w-3.5 h-3.5" /> Fleet Distance</span>
            <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${!activeMetrics ? 'bg-zinc-800/50 text-zinc-600' : percChangeDist < 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
              {activeMetrics ? (percChangeDist > 0 ? '+' : '') + percChangeDist + '%' : '-'}
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white font-mono tracking-tight">{metrics.distance.current}<span className="text-sm font-normal text-zinc-500 ml-0.5">{activeMetrics && 'km'}</span></span>
            <span className="text-xs text-zinc-500 font-mono line-through">{metrics.distance.baseline}{activeMetrics && 'km'}</span>
          </div>
        </div>

        <div className="space-y-1 pt-3 border-t border-zinc-800/50">
          <div className="flex justify-between items-center text-xs text-zinc-500 font-medium">
             <span className="flex items-center gap-1.5"><Zap className="w-3.5 h-3.5" /> Fleet Utilization</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-bold text-blue-400 font-mono tracking-tight">{metrics.utilization.current}</span>
            <span className="text-xs text-zinc-500 font-mono line-through">{metrics.utilization.baseline}</span>
          </div>
        </div>

        <div className="space-y-1 pt-3 border-t border-zinc-800/50">
          <div className="flex justify-between items-center text-xs text-zinc-500 font-medium">
             <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> SLA Breach Rate</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-bold text-emerald-400 font-mono tracking-tight">{metrics.sla.current}</span>
            <span className="text-xs text-zinc-500 font-mono line-through">{metrics.sla.baseline}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
 
