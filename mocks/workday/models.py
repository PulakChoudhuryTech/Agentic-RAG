"""Pydantic response models for the mock Workday API."""

from __future__ import annotations

from pydantic import BaseModel


class Benefits(BaseModel):
    health_plan: str
    dependents_covered: int


class Employee(BaseModel):
    employee_id: str
    name: str
    email: str
    country: str
    department: str
    manager_id: str
    hire_date: str
    tenure_months: int
    employment_type: str


class LeaveBalance(BaseModel):
    employee_id: str
    vacation_days_total: int
    vacation_days_used: int
    vacation_days_available: int


class BenefitsResponse(BaseModel):
    employee_id: str
    health_plan: str
    dependents_covered: int
