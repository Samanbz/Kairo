import { useCallback, useState} from "react";
import { CloudRain, Truck, UploadCloud,  CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "./ui/button";
import { useStore } from "../store/useStore";

function FileDropzone({ label, isUploaded, error, onUpload }: { label: string, isUploaded: boolean, error?: string | null, onUpload: (file: File) => void }) {
  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUpload(e.dataTransfer.files[0]);
    }
  }, [onUpload]);

  return (
    <div 
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      className={`border-2 border-dashed ${error ? 'border-red-500/50 bg-red-500/10' : isUploaded ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-zinc-800 bg-zinc-900/80'} rounded-xl p-5 text-center cursor-pointer hover:border-zinc-600 transition-colors flex flex-col items-center justify-center relative`}
      onClick={() => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = ".csv,.json";
        input.onchange = (e: any) => e.target.files && onUpload(e.target.files[0]);
        input.click();
      }}
    >
      {error ? <AlertCircle className="w-6 h-6 text-red-500 mb-2" /> : isUploaded ? <CheckCircle2 className="w-6 h-6 text-emerald-500 mb-2" /> : <UploadCloud className="w-6 h-6 text-zinc-500 mb-2" />}
      <span className="text-sm font-semibold text-zinc-300">
        {label}
      </span>
      <span className={`text-xs mt-1 ${error ? 'text-red-400' : 'text-zinc-500'}`}>
        {error ? error : isUploaded ? 'Ready for simulation' : 'Click or drop .CSV'}
      </span>
    </div>
  );
}

export function ControlSidebar() {
  const { 
    weather, setWeather,
    scenario, setScenario,
    ordersUploaded, setOrdersUploaded,
    vehiclesUploaded, setVehiclesUploaded,
    setTimeBounds, setSimTime
  } = useStore();

  const [ordersError, setOrdersError] = useState<string | null>(null);
  const [vehiclesError, setVehiclesError] = useState<string | null>(null);

  const handleOrdersUpload = async (f: File) => {
    try {
      setOrdersError(null);
      // Simulate backend delay/failure logic
      await new Promise((resolve) => {
        setTimeout(() => resolve(true), 500); 
      });
      console.log("Uploaded orders:", f.name);
      setOrdersUploaded(true);
      setTimeBounds([8.0, 18.5]); 
      setSimTime(8.0);
    } catch (e: any) {
      console.error(e);
      setOrdersError(e.message || "Failed to upload orders");
      setOrdersUploaded(false);
    }
  };

  const handleVehiclesUpload = async (f: File) => {
    try {
      setVehiclesError(null);
      // Simulate backend delay/failure logic
      await new Promise((resolve) => {
        setTimeout(() => resolve(true), 500);
      });
      console.log("Uploaded vehicles:", f.name);
      setVehiclesUploaded(true);
    } catch (e: any) {
      console.error(e);
      setVehiclesError(e.message || "Failed to upload vehicles");
      setVehiclesUploaded(false);
    }
  };

  return (
    <div className="flex flex-col h-full text-zinc-300 p-6 shadow-2xl overflow-y-auto w-80 shrink-0 border-r border-zinc-800 bg-zinc-950/95 backdrop-blur-md">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2 tracking-tight">
          <div className="bg-blue-600 p-1.5 rounded-lg text-white">
            <Truck className="w-5 h-5" />
          </div>
          Kairo OCC
        </h1>
        <p className="text-xs text-zinc-500 mt-2 font-medium tracking-wide uppercase">Operational Control Center</p>
      </div>

      {/* Files Input Zone */}
      <div className="space-y-4 mb-8">
        <h2 className="text-xs font-bold text-zinc-500 uppercase tracking-widest pl-1">Dataset Config</h2>
        <div className="flex flex-col gap-3">
          <FileDropzone label="Upload Orders" isUploaded={ordersUploaded} error={ordersError} onUpload={handleOrdersUpload} />
          <FileDropzone label="Upload Vehicles" isUploaded={vehiclesUploaded} error={vehiclesError} onUpload={handleVehiclesUpload} />
        </div>
      </div>

      {/* Model & Scenario Config */}
      <div className="space-y-4 mb-8">
        <h2 className="text-xs font-bold text-zinc-500 uppercase tracking-widest pl-1">Route & Strategy</h2>
        <div className="bg-zinc-900/80 border border-zinc-800 p-1.5 rounded-xl">
          <select 
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            className="bg-transparent text-zinc-200 font-medium w-full outline-none text-sm p-2 cursor-pointer focus:ring-2 focus:ring-blue-500/50 rounded-lg"
          >
            <option>Scenario B: Optimized Target</option>
            <option>Scenario A: Legacy Baseline</option>
            <option>Scenario C: Predictive Rush Hour</option>
          </select>
        </div>
      </div>

      {/* Environment Config */}
      <div className="space-y-4">
        <h2 className="text-xs font-bold text-zinc-500 uppercase tracking-widest pl-1">Environment Details</h2>
        <div className="bg-zinc-900/80 p-4 rounded-xl border border-zinc-800/80">
          <label className="flex items-center gap-2 text-sm text-zinc-300 font-medium mb-3">
            <CloudRain className="w-4 h-4 text-blue-400" /> Ground Weather Status
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(["Clear", "Rain", "Snow"] as const).map(w => (
              <Button 
                key={w} variant="outline" size="sm" 
                onClick={() => setWeather(w)}
                className={`py-1.5 h-auto text-xs font-semibold shadow-none border transition-all ${weather === w ? 'bg-blue-600 hover:bg-blue-500 text-white border-blue-500' : 'bg-transparent border-zinc-700 hover:bg-zinc-800 text-zinc-400'}`}
              >
                {w}
              </Button>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
