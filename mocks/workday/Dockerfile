FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic
WORKDIR /app
COPY mocks/workday /app/mocks/workday
COPY data/seed /app/data/seed
WORKDIR /app/mocks/workday
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
