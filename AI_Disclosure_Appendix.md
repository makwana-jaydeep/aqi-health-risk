# AI Disclosure Appendix

## Tool Used
Claude (Anthropic) — accessed via claude.ai

---

## Usage Summary

AI assistance was used strictly for boilerplate generation and documentation drafting.
All MLOps design decisions, tool selection, pipeline architecture, debugging, and
implementation logic were done independently.

---

## Specific Prompts and Sections

| Area | Prompt Summary | What Was Used |
|------|---------------|---------------|
| Project scaffolding | "Generate a Python project folder structure for a FastAPI backend with health endpoints, Pydantic schemas, and a services layer" | Initial file/folder layout only. All logic was written and debugged manually. |
| Docker Compose template | "Generate a docker-compose.yml template with FastAPI, Streamlit, Postgres, and Prometheus services" | Used as a starting template. All service configs, volume mounts, env vars, and health checks were modified manually during debugging. |
| Streamlit UI skeleton | "Generate a basic two-page Streamlit app with a sidebar and a form" | UI skeleton only. Page logic, API integration, and chart components written manually. |
| Documentation | "Draft an architecture document for a microservice system with these components" | First draft of docs/architecture.md, HLD.md, LLD.md, and user_manual.md. All content was reviewed and edited to match actual implementation. |
| Report writing | "Format this project information into a structured PDF report" | Final project report structure and formatting only. All technical content reflects actual work done. |

---

## What Was NOT Done by AI

- MLOps pipeline design (Airflow DAG topology, drift detection strategy, retraining trigger logic)
- DVC pipeline definition and stage dependencies
- MLflow experiment tracking configuration and feature importance logging
- Prometheus instrumentation decisions (which metrics to expose and why)
- Grafana alert rule configuration and SMTP setup
- All debugging (psycopg2, mlflow run conflicts, Airflow path issues, Grafana provisioning errors)
- Model selection rationale and hyperparameter choices
- Data generation logic and risk labelling logic
- All fixes applied during iterative development and testing

---

*Submitted by: Jaydeep Makwana — da25m013*
