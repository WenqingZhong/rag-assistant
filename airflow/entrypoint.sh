#!/bin/bash
set -e

echo "Initializing Airflow database..."
airflow db migrate

echo "Starting Airflow (standalone mode)..."
exec airflow standalone