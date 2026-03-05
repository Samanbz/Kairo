cd backend
conda activate kairo

pip install ./

fastapi dev main.py &

cd ../frontend

pnpm install

pnpm run dev