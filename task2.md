## 1. Архитектура витрин

```bash
Airflow DAG ежедневный
    ├── extract_from_sources -> ClickHouse raw
    ├── dbt run --select staging+               # staging слой
    ├── dbt run --select core+                  # core слой
    ├── dbt run --select marts+                 # mart слой
    └── dbt test                                # запуск всех тестов
```

**Staging** (stg_offers.sql) - слой сырых данных полученных из из различных источников, в нашем случае Clickhouse. Модели в dbt - это SELECT-запросы к сырым таблицам. Цель - создать "чистую" копию сырых данных.

**Core / Intermediate** (int_offers.sql / core_offers.sql) - основной слой преобразований. Здесь происходит обработка пропусков, удаление дублей, исправление единиц измерений, обработка выбросов,  создаются ключи для соединения таблиц.

**Mart** (marts_) - это финальные таблицы с отобранными, очищенными и агрегированными фичами, соединенными в один датасет.


```bash
commercial_offers/
├── README.md
├── dbt_project.yml
├── macros/
│   ├── standardize_units.sql    # Приведение единиц измерения
│   └── handle_outliers.sql      # Фильтрация/ограничение выбросов
├── models/
│   ├── staging/
│   │   ├── offers.yml           # Описание и тесты для staging
│   │   └── stg_offers.sql       # SELECT *
│   │
│   ├── core/
│   │   ├── core.yml             # Описание и тесты для core-слоя
│   │   └── core_offers.sql      # Дедупликация, импутация, приведение единиц
│   │
│   └── marts/
│       ├── marts.yml            # Описание и тесты фич
│       └── offer_features.sql   # Финальные фичи для ML
│
├── seeds/                       
│   └── unit_conversion.csv      # Вспомогательные csv для dbt проекта
│
├── packages.yml
├── snapshots
└── tests/                       # Пользовательские тесты
    └── test_no_extreme_outliers.py
```

## 2. Контроль качества данных (DQ в dbt)

Используются dbt tests на уровне моделей. Tests позволяют проверять качество и целостность данных в моделях. Они помогают гарантировать, что данные в вашей модели соответствуют определенным критериям. В dbt присутствует 3 вида тестов:

 1. **Singular**. Выполняет запрос к таблице на поиск строк с ошибкой.

 2. **Generic**. Это готовые тесты. Unique – колонка должна содержать уникальные значения, not_null колонка не должна содержать пустые значения (NULL), accepted_values – колонка должна содержать только значения из заданного набора, relationships – значения в колонке должны полностью соответствуют значениям в другой колонке.

 3. **Unit-тесты**. Позволяют проверить, что наша логика трансформации написана верно.
Уникальность ключей: unique, not_null.

```yaml
version: '1.0'

models:
  - name: core_offers
    columns:
      - name: price
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 1000000
      - name: currency
        tests:
          - accepted_values:
              values: ['USD', 'EUR', 'RUB']
      - name: client_id
        tests:
          - relationships:
              to: ref('stg_clients')
              field: id
```

## 3. Версионирование и документирование

Весь код (Airflow DAGs, dbt-модели, SQL-скрипты) хранится в Git. Каждое изменение, добавление фичи, изменение логики - это новый коммит и потенциально новая версия витрины. Используются теги для маркировки релизов.

Dbt генерирует документацию автоматически из schema.yml и .sql файлов. Описываются модели, их назначение, столбцы и тесты. Доступна через dbt docs generate и dbt docs serve. Dbt автоматически строит DAG зависимостей между staging -> core -> mart.

## 4. Обновление данных


1. Инкрементальная загрузка для staging слоя новых и измененных записей с использованием `ReplacingMergeTree` и поля версии (`_version`) для дедупликации.

2. Инкрементальное обновление core и marts слоёв через dbt с использованием:
   - `incremental` с `delete+insert` для замены данных за конкретный день
   - для таблиц с историей изменений используем поля `valid_from`/`valid_to`
   - `ALTER TABLE ... REPLACE PARTITION` только для полного пересчета исторических периодов

3. Для итоговых таблиц используем `ReplacingMergeTree` или материализованные представления (`MATERIALIZED VIEW`) с `FINAL` для автоматической дедупликации при чтении.

4. Основной пайплайн - ежедневный batch. Критические исправления - по требованию через ручной запуск.