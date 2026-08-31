FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic
WORKDIR /app
COPY mocks/servicenow /app/mocks/servicenow
RUN mkdir -p /app/mocks/servicenow/data
WORKDIR /app/mocks/servicenow
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
