FROM python:3.11-slim
RUN pip install --no-cache-dir streamlit httpx
WORKDIR /app
COPY ui /app/ui
EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
