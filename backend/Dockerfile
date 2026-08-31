FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -e .
COPY backend /app/backend
COPY db /app/db
COPY data /app/data
EXPOSE 8000
CMD ["sh", "-c", "python db/migrate.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
