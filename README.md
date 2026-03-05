# Kairo Digital Twin Project

## Setup Instructions

### Prerequisites
- Docker & Docker Compose
- Node.js (for local frontend dev)
- Python 3.10+ (for local backend dev)

### Quick Start
1. Build and run the stack:
   ```bash
   docker-compose up --build
   ```

2. Access the application:
   - Frontend: [http://localhost:5173](http://localhost:5173)
   - Backend API: [http://localhost:8000](http://localhost:8000)
   - API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Development

#### Backend
```bash
cd backend
pip install -r requirements.txt # or pip install .
uvicorn src.main:app --reload
```
**Linting:**
```bash
ruff check .
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```
**Linting:**
```bash
npm run lint
npm run format
```


Setup

first get city's map using `fetch_map.py`, e.g. for Darmstadt, using exact coordinates.
then
```
netconvert --osm-files sumo_data/map.osm \
  -o sumo_data/darmstadt.net.xml \
  --geometry.remove \
  --roundabouts.guess \
  --ramps.guess \
  --junctions.join \
  --tls.guess-signals \
  --tls.discard-simple \
  --tls.join
```
```
python /usr/share/sumo/tools/randomTrips.py \
    -n sumo_data/darmstadt.net.xml \
    -o sumo_data/trips.trips.xml \
    -e 3600 \
    --period 2.0
```
```
duarouter \
    -n sumo_data/darmstadt.net.xml \
    --route-files sumo_data/trips.trips.xml \
    -o sumo_data/routes.rou.xml \
    --ignore-errors
```

create file backend/run.sumcfg
```
<configuration>
    <input>
        <net-file value="./sumo_data/darmstadt.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
```