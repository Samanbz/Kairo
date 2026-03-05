import { create } from 'zustand';

interface MapState {
  weather: 'Clear' | 'Rain' | 'Snow';
  setWeather: (w: 'Clear' | 'Rain' | 'Snow') => void;
  scenario: string;
  setScenario: (s: string) => void;
  
  ordersUploaded: boolean;
  setOrdersUploaded: (val: boolean) => void;
  vehiclesUploaded: boolean;
  setVehiclesUploaded: (val: boolean) => void;

  isRunning: boolean;
  setIsRunning: (val: boolean) => void;

  simTime: number;
  setSimTime: (val: number) => void;

  timeBounds: [number, number];
  setTimeBounds: (val: [number, number]) => void;

  baselineStrategy: string;
  setBaselineStrategy: (val: string) => void;

  activeMetrics: any | null;
  setActiveMetrics: (metrics: any) => void;

  isReadyToSimulate: () => boolean;
}

export const useStore = create<MapState>((set, get) => ({
  weather: 'Clear',
  setWeather: (w) => set({ weather: w }),
  scenario: 'Scenario B: Optimized Target',
  setScenario: (s) => set({ scenario: s }),

  ordersUploaded: false,
  setOrdersUploaded: (val) => set({ ordersUploaded: val }),
  vehiclesUploaded: false,
  setVehiclesUploaded: (val) => set({ vehiclesUploaded: val }),

  isRunning: false,
  setIsRunning: (val) => {
    set({ isRunning: val });
    if(val && !get().activeMetrics) {
      // Mock generating metrics when simulation starts for the first time
      set({
        activeMetrics: {
          distance: { baseline: 145, current: 120 },
          utilization: { baseline: '60%', current: '88%' },
          sla: { baseline: '88%', current: '99%' }
        }
      });
    }
  },

  simTime: 0,
  setSimTime: (val) => set({ simTime: val }),

  timeBounds: [0.0, 24.0],
  setTimeBounds: (val) => set({ timeBounds: val }),

  baselineStrategy: 'Compare vs Legacy Baseline',
  setBaselineStrategy: (val) => set({ baselineStrategy: val }),

  activeMetrics: null,
  setActiveMetrics: (metrics) => set({ activeMetrics: metrics }),

  isReadyToSimulate: () => {
    const s = get();
    return s.ordersUploaded && s.vehiclesUploaded;
  }
}));
