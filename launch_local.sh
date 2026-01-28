cd backend
conda activate kairo

pip install ./ -e

fastapi dev main.py &

cd ../frontend

pnpm install

pnpm run dev