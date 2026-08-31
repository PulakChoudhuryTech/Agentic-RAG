import httpx

from backend.app.config import get_settings


def test_get_employee(services_available):
    settings = get_settings()
    resp = httpx.get(f"{settings.workday_api_url}/employees/E1001")
    resp.raise_for_status()
    data = resp.json()
    assert data["employee_id"] == "E1001"
    assert data["country"] == "IN"


def test_get_leave_balance(services_available):
    settings = get_settings()
    resp = httpx.get(f"{settings.workday_api_url}/employees/E1002/leave-balance")
    resp.raise_for_status()
    data = resp.json()
    assert data["vacation_days_available"] == 15


def test_get_benefits(services_available):
    settings = get_settings()
    resp = httpx.get(f"{settings.workday_api_url}/employees/E1001/benefits")
    resp.raise_for_status()
    assert resp.json()["health_plan"] == "Premium"


def test_unknown_employee_returns_404(services_available):
    settings = get_settings()
    resp = httpx.get(f"{settings.workday_api_url}/employees/DOES_NOT_EXIST")
    assert resp.status_code == 404
