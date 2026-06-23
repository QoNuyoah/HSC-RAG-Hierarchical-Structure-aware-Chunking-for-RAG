# HSC-RAG Dashboard Usage

## Start Backend

```powershell
cd /d E:\practical_training\HSC_RAG\backend
& E:\anaconda3\envs\HSC_RAG\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

## Start Frontend

```powershell
cd /d E:\practical_training\HSC_RAG\frontend
npm.cmd run dev -- --port 5173
```

Dashboard:

```text
http://127.0.0.1:5173
```

## Main API Endpoints

```text
GET /api/overview
GET /api/metrics?retriever=bm25
GET /api/metrics?retriever=dense
GET /api/metrics?retriever=hybrid
GET /api/queries?retriever=bm25
GET /api/bad-cases?retriever=bm25
GET /api/queries/{query_id}/comparison?retriever=bm25
POST /api/cache/refresh
```

## Dashboard Scope

The dashboard reads existing experiment artifacts under:

```text
data\processed\qasper\train
```

It does not recompute chunking or retrieval. After rerunning chunking/retrieval scripts, call:

```text
POST /api/cache/refresh
```

or restart the backend server.

