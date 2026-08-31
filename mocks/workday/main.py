"""
Mock Workday API.

Stands in for a real HR information system. Deliberately a separate FastAPI
process (see docker-compose.yml, ports 8001) rather than routes inside the
main backend app, so the agent code in backend/app/clients/workday_client.py
has to make a genuine HTTP call -- with the timeouts/retries that implies --
exactly like it would against a real Workday tenant.

Endpoints:
  GET /employees/{employee_id}
  GET /employees/{employee_id}/leave-balance
  GET /employees/{employee_id}/benefits
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from models import BenefitsResponse, Employee, LeaveBalance
from seed_data import load_employees

app = FastAPI(title="Mock Workday API")

EMPLOYEES = load_employees()


def _get_employee_or_404(employee_id: str) -> dict:
    employee = EMPLOYEES.get(employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail=f"employee {employee_id} not found")
    return employee


@app.get("/employees/{employee_id}", response_model=Employee)
def get_employee(employee_id: str) -> Employee:
    e = _get_employee_or_404(employee_id)
    return Employee(
        employee_id=e["employee_id"],
        name=e["name"],
        email=e["email"],
        country=e["country"],
        department=e["department"],
        manager_id=e["manager_id"],
        hire_date=e["hire_date"],
        tenure_months=e["tenure_months"],
        employment_type=e["employment_type"],
    )


@app.get("/employees/{employee_id}/leave-balance", response_model=LeaveBalance)
def get_leave_balance(employee_id: str) -> LeaveBalance:
    e = _get_employee_or_404(employee_id)
    return LeaveBalance(
        employee_id=e["employee_id"],
        vacation_days_total=e["vacation_days_total"],
        vacation_days_used=e["vacation_days_used"],
        vacation_days_available=e["vacation_days_available"],
    )


@app.get("/employees/{employee_id}/benefits", response_model=BenefitsResponse)
def get_benefits(employee_id: str) -> BenefitsResponse:
    e = _get_employee_or_404(employee_id)
    return BenefitsResponse(
        employee_id=e["employee_id"],
        health_plan=e["benefits"]["health_plan"],
        dependents_covered=e["benefits"]["dependents_covered"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "employees_loaded": len(EMPLOYEES)}
