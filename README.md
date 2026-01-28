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
