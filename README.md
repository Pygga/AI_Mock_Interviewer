# AI Mock Interviewer 🎓🤖

Агентный тренажер технических собеседований для Python-разработчиков.  

---

## 📌 Архитектура ДЗ №1 (Модуль 01)

* **Инференс:** Локальный запуск компактных моделей ($\le 4\text{B}$) через `llama-server` с аппаратным ускорением Apple Metal.
* **Протокол вызова функций (Tools):** Без использования нативного Function Calling/MCP. Агент генерирует маркеры `$call$func_name(args)$call$`, которые парсятся диспетчером Python.
* **Автономный цикл (ReAct):** Результат выполнения локальной функции сразу возвращается в контекст модели без ожидания ввода от пользователя.
* **Логирование:** Сохранение полной стенограммы сессии в файлы `interview_<model>_<timestamp>.log`.

---

## 🛠️ Структура проекта

```text
AI_Mock_Interviewer/
├── models/                  # Директория для локальных GGUF-моделей (в .gitignore)
├── interview_agent.py       # Основной консольный REPL-агент (чистый Python, без pip-зависимостей)
├── .gitignore               # Игнорирование весов моделей и логов
└── README.md                # Инструкция и описание проекта
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