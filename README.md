## Run example
uvicorn api2:app --reload --port 8001

### POST example

http://localhost:8001/predict

body:
{
  "symptoms": [

    "fatigue",
    "dizziness",
    "weakness",
    "depression",
    "anxiety and nervousness"
  ]
}
