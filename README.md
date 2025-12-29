
```bash
diabetes-prediction/
├── README.md
├── requirements.txt
├── Dockerfile
├── train.py                     # скрипт обучения с MLflow
├── predict.py                   # скрипт предсказания (CLI)
├── app/
│   └── main.py                  # FastAPI сервис
├── src/
│   ├── data.py                  # загрузка, валидация, очистка
│   ├── features.py              # feature engineering
│   └── model.py                 # обучение CatBoost/LightGBM
├── tests/
│   ├── test_api.py              # тест /predict
│   ├── test_model_io.py         # проверка формата вход/выход
│   └── test_regression.py       # проверка, что скор не упал
├── data/
│   ├── train.csv
│   └── test.csv
└── models/                      # (будет создан MLflow)
```