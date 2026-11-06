# AI Mock Interviewer 🎓🤖

Агентный тренажер технических собеседований для Python-разработчиков.  

---

## 📌 Архитектура ДЗ №1 (Модуль 01)

* **Инференс:** Локальный запуск компактных моделей ($\le 4\text{B}$) через `llama-server` с аппаратным ускорением Apple Metal.
* **Протокол вызова функций (Tools):** Без использования нативного Function Calling/MCP. Агент генерирует маркеры `$call$func_name(args)$call$`, которые парсятся диспетчером Python.
* **Автономный цикл (ReAct):** Результат выполнения локальной функции сразу возвращается в контекст модели без ожидания ввода от пользователя.
* **Логирование:** Сохранение полной стенограммы сессии в файлы `interview_<model>_<timestamp>.log`.
* **Отчет:** `LAB_01_CONCLUSIONS.md`.

---

## 📌 Архитектура ДЗ №2 (Модули 02–03)

* **Фреймворк:** `LangChain` на базе LCEL (`prompt | llm | parser | custom_cell`).
* **Память фактов (Fact Memory):** Долговременное хранение профиля кандидата и статуса вопросов в файле `memory.json` (вместо тяжелой истории сообщений).
* **Собственная ячейка (Custom CELL):** Изолированный вывод служебных данных (оценка, тема, синхронизация памяти) отдельно от реплик интервьюера.
* **Непрерывность сессий (Resume):** При перезапуске агент считывает `memory.json`, пропускает уже отвеченные вопросы и сразу продолжает диалог с оставшихся.
* **Отчет:** `LAB_02_CONCLUSIONS.md`.

---

## 📌 Архитектура ДЗ №3 (Модули 04–05)

* **Гибридная память (3 уровня):** Оконная память последних реплик (`BufferWindowMemory`, $k=2$) + суммаризирующая память (`SummaryMemory` через LLM) + файловая память фактов и уровня в `memory_v3.json`.
* **Условное ветвление (`RunnableBranch`):** Классификация реплики через `PydanticOutputParser` (`intent`, `sentiment`, `has_code`) и маршрутизация по 5 веткам (эмпатия при усталости/негативе, выдача подсказки, оценка уровней Junior / Middle / Senior).
* **Адаптивная сложность:** Динамическое повышение или понижение уровня кандидата (`junior` $\leftrightarrow$ `middle` $\leftrightarrow$ `senior`) в зависимости от оценок.
* **Инструменты `@tool`:** Набор типизированных инструментов (`fetch_question_by_level`, `get_topic_hint`, `check_python_syntax` с AST-разбором кода, `record_score_and_adapt_level`).
* **Отчет:** `LAB_03_CONCLUSIONS.md`.

---

## 🛠️ Структура проекта

```text
AI_Mock_Interviewer/
├── models/                    # Локальные GGUF-модели (в .gitignore)
├── interview_agent.py         # [ДЗ-1] Консольный агент на чистом Python (ручной парсинг tools)
├── interview_agent_v2.py      # [ДЗ-2] Агент на LangChain LCEL с памятью фактов и Custom CELL
├── memory.json                # [ДЗ-2] Файл персистентной памяти фактов кандидата
├── interview_agent_v3.py      # [ДЗ-3] Агент с Window+Summary памятью, RunnableBranch и @tool
├── memory_v3.json             # [ДЗ-3] Гибридная память (уровень знаний, окно k=2, сводка, банк тем)
├── LAB_01_CONCLUSIONS.md      # Итоговый отчет и сравнение моделей по ДЗ-1
├── LAB_02_CONCLUSIONS.md      # Итоговый отчет по архитектуре и тестам ДЗ-2
├── LAB_03_CONCLUSIONS.md      # Итоговый отчет по ветвлению, памяти и инструментам ДЗ-3
├── .gitignore                 # Игнорирование весов моделей и логов
└── README.md                  # Документация проекта
```

---

## 🚀 Быстрый запуск

### 1. Перемещение моделей в проект
Поместите скачанные файлы `.gguf` в папку `models/`:
* `models/qwen2.5-3b-instruct-q4_k_m.gguf`
* `models/Llama-3.2-3b-instruct-q4_k_m.gguf`

### 2. Запуск локального инференса (Терминал 1)
```bash
# Для модели Qwen 2.5:
llama-server -m models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -c 4096 --port 8080

# Для модели Llama 3.2:
llama-server -m models/Llama-3.2-3b-instruct-q4_k_m.gguf -ngl 99 -c 4096 --port 8080
```

### 3. Запуск агента-интервьюера (Терминал 2)
```bash
# Прогон на модели Qwen:
python3 interview_agent.py qwen

# Прогон на модели Llama:
python3 interview_agent.py llama
```

---

## 🧩 Реализованные инструменты (Tools)

1. `get_question(topic, level)` — получение вопроса из базы знаний (Python, SQL, Algorithms).
2. `check_code(task_name, code)` — эмуляция проверки решения на тест-кейсах.
3. `record_score(topic, score)` — сохранение балла в протокол.
4. `finish_interview()` — подведение итогов и формирование финального вердикта.