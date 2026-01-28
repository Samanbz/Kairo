import { create } from 'zustand'
import { useQuery } from '@tanstack/react-query'

// Zustand Store
interface AppState {
  count: number
  increment: () => void
}

const useStore = create<AppState>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}))

// Fetch function
const fetchHello = async () => {
  try {
    const res = await fetch('http://localhost:8000/')
    if (!res.ok) throw new Error('Network response was not ok')
    return res.json()
  } catch (error) {
    console.error("Fetch error:", error)
    return { message: "Backend not connected (Check console)" }
  }
}

function App() {
  const { count, increment } = useStore()

  const { data, isLoading } = useQuery({
    queryKey: ['hello'],
    queryFn: fetchHello
  })

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-8 space-y-6">
        <h1 className="text-3xl font-bold text-gray-900 text-center">
          Kairo Digital Twin
        </h1>
        
        <div className="bg-blue-50 p-4 rounded-md border border-blue-200">
          <h2 className="text-lg font-semibold text-blue-800 mb-2">Backend Status</h2>
          {isLoading ? (
            <p className="text-blue-600">Loading...</p>
          ) : (
            <p className="text-blue-700 font-mono text-sm">
              {JSON.stringify(data, null, 2)}
            </p>
          )}
        </div>

        <div className="bg-green-50 p-4 rounded-md border border-green-200">
          <h2 className="text-lg font-semibold text-green-800 mb-2">Zustand Store</h2>
          <div className="flex items-center justify-between">
            <span className="text-green-700">Count: {count}</span>
            <button 
              onClick={increment}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
            >
              Increment
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
