# ML Pipeline предсказания развития диабета

End-to-end ML пайплайн для прогнозирования риска развития диабета на основе данных о состоянии здоровья и образе жизни.

## Запуск

### Обучение модели

```bash
python train.py
```

### Инференс локально

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Инференс с помощь docker

```bash
docker build -t model-api .
docker run -p 8000:8000 model-api
```

### Запуск тестов
```bash
PYTHONPATH=. pytest tests/ -v
```

## API

### Request Example

```json
{
  "age": 45,
  "alcohol_consumption_per_week": 2,
  "physical_activity_minutes_per_week": 150,
  "diet_score": 7.5,
  "sleep_hours_per_day": 7.0,
  "screen_time_hours_per_day": 4.0,
  "bmi": 28.5,
  "waist_to_hip_ratio": 0.9,
  "systolic_bp": 120,
  "diastolic_bp": 80,
  "heart_rate": 72,
  "cholesterol_total": 200,
  "hdl_cholesterol": 50,
  "ldl_cholesterol": 130,
  "triglycerides": 150,
  "gender": "Male",
  "ethnicity": "Caucasian",
  "education_level": "Bachelor",
  "income_level": "Middle",
  "smoking_status": "Never",
  "employment_status": "Employed",
  "family_history_diabetes": 1,
  "hypertension_history": 0,
  "cardiovascular_history": 0
}
```

```json
{
  "diabetes_probability": 1.0
}
```

## Структура проекта
```bash
diabetes-prediction/

├── app/
│   └── main.py                  # FastAPI сервис инференса
│
├── data/
│   ├── train.csv                # Датасет для обучения
│   └── test.csv                 # Тестовый датасет
│
├── src/
│   ├── data.py                  # Загрузка, валидация, очистка
│   ├── features.py              # Feature engineering
│   └── model.py                 # Обучение CatBoost модели
│
├── tests/
│   ├── test_api.py              # Тест эндпоинта /predict
│   ├── test_model_io.py         # Проверка формата вход/выход
│   └── test_regression.py       # Проверка, что скор не упал
│
├── Dockerfile                   # Докер образ
├── README.md                    # Описание проекта
├── requirements.txt             # Зависимости
├── task1.md                     # Задание 1
├── task2.md                     # Задание 2
└── train.py                     # Скрипт обучения модели с MLflow
```

