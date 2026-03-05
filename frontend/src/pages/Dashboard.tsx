import { MapCanvas } from "../components/MapCanvas";
import { ControlSidebar } from "../components/ControlSidebar";
import { KPIDashboard } from "../components/KPIDashboard";

export function Dashboard() {
  return (
    <div className="relative w-full h-screen flex bg-zinc-950 font-sans">
      <div className="h-full z-10 border-r border-zinc-800/80 bg-zinc-950 shadow-xl max-w-sm">
        <ControlSidebar />
      </div>
      
      <div className="relative flex-1 h-screen overflow-hidden">
        <MapCanvas />
        <div className="absolute top-6 right-6 z-20 pointer-events-none">
          <KPIDashboard />
        </div>
      </div>
    </div>
  );
}
