.PHONY: install db-up db-down migrate ingest run-backend run-ui run-workday run-servicenow test eval-retrieval eval-agents eval-ragas eval-all fmt

install:
	pip install -e ".[dev,eval]"

db-up:
	docker compose up -d postgres workday_mock servicenow_mock

db-down:
	docker compose down

migrate:
	python db/migrate.py

ingest:
	python -m backend.app.ingestion.ingest

run-backend:
	uvicorn backend.app.main:app --reload --port 8000

run-ui:
	streamlit run ui/streamlit_app.py

run-workday:
	cd mocks/workday && uvicorn main:app --reload --port 8001

run-servicenow:
	cd mocks/servicenow && uvicorn main:app --reload --port 8002

test:
	pytest backend/tests eval/test_eval_thresholds.py -v

eval-retrieval:
	python -m eval.run_eval --suite retrieval

eval-agents:
	python -m eval.run_eval --suite agents

eval-ragas:
	python -m eval.run_eval --suite ragas

eval-all:
	python -m eval.run_eval --suite all
