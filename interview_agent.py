#!/usr/bin/env python3
"""
AI Mock Interviewer — Агент технического собеседования (ДЗ 01).
Версия 1.1: Исправлено зацикливание, добавлена поддержка нескольких вызовов $call$
и корректное ожидание ответа кандидата после каждого вопроса.
"""

import os
import sys
import re
import json
import urllib.request
import urllib.error
import datetime

# ---------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ
# ---------------------------------------------------------------------------
API_URL = os.getenv("LLAMA_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "local-model")

log_label = sys.argv[1] if len(sys.argv) > 1 else "model"
LOG_FILENAME = f"interview_{log_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def write_log(text: str):
    """Запись событий сессии в файл."""
    with open(LOG_FILENAME, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def send_chat_completion(messages: list) -> str:
    """Запрос к локальному llama-server через urllib (без pip-зависимостей)."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"] or ""


# ---------------------------------------------------------------------------
# 2. БАЗА ВОПРОСОВ И ИНСТРУМЕНТЫ (TOOLS)
# ---------------------------------------------------------------------------
QUESTIONS_DB = {
    ("python", "junior"): "В чем разница между списками (list) и кортежами (tuple) в Python? Когда стоит выбрать кортеж?",
    ("python", "middle"): "Как работает GIL (Global Interpreter Lock) в CPython? В каких задачах он становится узким местом, а когда не влияет?",
    ("python", "senior"): "Опиши внутреннее устройство словаря (dict) в CPython 3.7+ и как достигается сложность O(1) при сохранении порядка вставки.",
    ("sql", "junior"): "В чем разница между операторами WHERE и HAVING в SQL?",
    ("sql", "middle"): "Что такое B-Tree индекс и почему поиск по условию LIKE '%текст' не может эффективно использовать индекс?",
    ("algorithms", "middle"): "Напиши функцию на Python, проверяющую сбалансированность скобок в строке '()[]{}'. Какова сложность твоего алгоритма?",
    ("orm", "middle"): "Что такое проблема N+1 запросов в ORM (например, Django ORM / SQLAlchemy) и как она решается?"
}

CANDIDATE_SCORES = {}


def tool_get_question(topic: str, level: str) -> str:
    topic = topic.strip().lower()
    level = level.strip().lower()
    # Нормализация уровня
    if level in ["intermediate", "mid"]:
        level = "middle"
    if level in ["beginner", "jun"]:
        level = "junior"

    key = (topic, level)
    if key in QUESTIONS_DB:
        return QUESTIONS_DB[key]
    
    # Поиск хотя бы по теме
    for (t, l), q in QUESTIONS_DB.items():
        if t == topic:
            return q

    return f"Расскажи про ключевые особенности и лучшие практики при работе с {topic.upper()} в Python."


def tool_check_code(task_name: str, code_snippet: str) -> str:
    code_clean = code_snippet.strip()
    if len(code_clean) < 15:
        return "Тесты не пройдены: решение слишком короткое или содержит синтаксическую ошибку."
    return "Тесты успешно пройдены (3/3). Временная сложность O(N), память O(N). Решение принято!"


def tool_record_score(topic: str, score_points: str) -> str:
    try:
        pts = int(re.search(r"\d+", score_points).group())
    except Exception:
        pts = 8
    pts = max(1, min(10, pts))
    CANDIDATE_SCORES[topic.strip()] = pts
    return f"Оценка за тему '{topic.strip()}' сохранена: {pts}/10."


def tool_finish_interview() -> str:
    if not CANDIDATE_SCORES:
        total = 8.0
        details = "Базовый уровень подтвержден."
    else:
        total = round(sum(CANDIDATE_SCORES.values()) / len(CANDIDATE_SCORES), 1)
        details = ", ".join(f"{k}: {v}/10" for k, v in CANDIDATE_SCORES.items())
    
    return f"Итоги собеседования: средний балл {total}/10. Оценки по темам: {details}. Вердикт: Рекомендован к офферу."


# ---------------------------------------------------------------------------
# 3. PROMPT ENGINEERING (Лекция 03_promt_eng.html)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Ты — Senior Tech Lead и профессиональный технический интервьюер по Python.
Твоя цель: провести первичное собеседование с кандидатом.

ГЛАВНЫЕ ПРАВИЛА:
1. Задавай строго ОДИН вопрос за раз.
2. Когда тебе нужен вопрос, вызови $call$get_question(тема, уровень)$call$.
3. Когда кандидат ответил на вопрос:
   - Кратко прокомментируй его ответ (похвали или поправь).
   - Вызови команду $call$record_score(тема, балл_от_1_до_10)$call$.
   - Запроси следующий вопрос через $call$get_question(тема, уровень)$call$.
4. Если кандидат пишет "завершить" — вызови $call$finish_interview()$call$ и подведи итоги.

ФОРМАТ ВЫЗОВОВ:
$call$get_question(python, middle)$call$
$call$record_score(python, 9)$call$
$call$finish_interview()$call$

ПРИМЕР ДИАЛОГА:
Кандидат: Привет, готов к собеседованию Middle.
Интервьюер: Привет! Начнем с теории Python. $call$get_question(python, middle)$call$
"""

# ---------------------------------------------------------------------------
# 4. ОБРАБОТЧИК ВЫЗОВОВ
# ---------------------------------------------------------------------------
def parse_and_execute_tools(text: str):
    """
    Находит ВСЕ вызовы $call$...$call$ в тексте ответа, выполняет их
    и возвращает список результатов и флаг необходимости ожидания пользователя.
    """
    calls = re.findall(r"\$call\$(.*?)\$call\$", text)
    if not calls:
        return [], None, False

    executed = []
    question_to_ask = None
    is_finished = False

    for raw in calls:
        raw = raw.strip()
        try:
            func_name = raw.split("(")[0].strip()
            args_part = raw.split("(", 1)[1].rsplit(")", 1)[0]
            args = [a.strip().strip("'\"") for a in args_part.split(",") if a.strip()]

            if func_name == "get_question":
                topic = args[0] if len(args) > 0 else "python"
                lvl = args[1] if len(args) > 1 else "middle"
                q = tool_get_question(topic, lvl)
                question_to_ask = (topic, q)
                executed.append(f"get_question({topic}) -> '{q}'")

            elif func_name == "record_score":
                topic = args[0] if len(args) > 0 else "general"
                pts = args[1] if len(args) > 1 else "8"
                res = tool_record_score(topic, pts)
                executed.append(res)

            elif func_name == "check_code":
                t_name = args[0] if len(args) > 0 else "task"
                code = args[1] if len(args) > 1 else ""
                res = tool_check_code(t_name, code)
                executed.append(res)

            elif func_name == "finish_interview":
                res = tool_finish_interview()
                executed.append(res)
                is_finished = True

        except Exception as e:
            executed.append(f"Ошибка вызова {raw}: {e}")

    return executed, question_to_ask, is_finished


def clean_display_text(text: str) -> str:
    """Удаляет технические маркеры $call$...$call$ из текста для чистого отображения кандидату."""
    cleaned = re.sub(r"\$call\$.*?\$call\$", "", text).strip()
    return cleaned


# ---------------------------------------------------------------------------
# 5. ОСНОВНОЙ REPL ЦИКЛ
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  AI Mock Interviewer — Терминал Собеседования (v1.1)")
    print(f"  API:        {API_URL}")
    print(f"  Лог-файл:   {LOG_FILENAME}")
    print("  Выход:      'выход', 'exit', 'конец', 'quit', 'завершить'")
    print("=" * 65)

    write_log(f"=== ЗАПУСК СЕССИИ: {datetime.datetime.now()} ===")
    write_log(f"API_URL: {API_URL}")
    write_log(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n" + "-" * 50)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("\n\033[1;32mКандидат > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nСессия прервана.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["выход", "exit", "конец", "quit"]:
            print("\nЗавершение собеседования. До свидания!")
            write_log("\n[СЕССИЯ ЗАВЕРШЕНА ПОЛЬЗОВАТЕЛЕМ]")
            break

        messages.append({"role": "user", "content": user_input})
        write_log(f"\nКандидат: {user_input}")

        # Запрос к LLM
        try:
            assistant_raw = send_chat_completion(messages)
        except urllib.error.URLError as err:
            print(f"\n\033[1;31m[Ошибка подключения к llama-server]: {err}\033[0m")
            print("Убедитесь, что в первом терминале запущен: llama-server --port 8080")
            break
        except Exception as err:
            print(f"\n\033[1;31m[Ошибка генерации]: {err}\033[0m")
            break

        # Добавляем сырой ответ в историю сообщений
        messages.append({"role": "assistant", "content": assistant_raw})
        write_log(f"Интервьюер (сырой): {assistant_raw}")

        # Парсим вызовы инструментов
        tool_results, question_info, is_finished = parse_and_execute_tools(assistant_raw)

        # Выводим текстовую часть реплики интервьюера (без тегов $call$)
        spoken_text = clean_display_text(assistant_raw)
        if spoken_text:
            print(f"\033[1;36mИнтервьюер >\033[0m {spoken_text}")

        # Показываем статус выполнения инструментов
        for res in tool_results:
            if "Оценка" in res:
                print(f"\033[0;33m  📊 [{res}]\033[0m")
                write_log(f"Система: {res}")
            elif "Итоги" in res:
                print(f"\n\033[1;32m🏆 [{res}]\033[0m")
                write_log(f"Система: {res}")

        # Если был запрошен вопрос — озвучиваем его кандидату
        if question_info:
            topic, q_text = question_info
            print(f"\n\033[1;37m❓ [Вопрос по {topic.upper()}]: {q_text}\033[0m")
            write_log(f"Вопрос кандидату: {q_text}")
            # Добавляем системное примечание в контекст, чтобы модель знала, какой вопрос был задан
            messages.append({
                "role": "system",
                "content": f"[ИНФОРМАЦИЯ]: Кандидату задан вопрос: '{q_text}'. Жди его ответа и оцени его."
            })

        if is_finished:
            print("\n\033[1;32m[Собеседование завершено! Полный отчет сохранен в лог-файл]\033[0m")
            break


if __name__ == "__main__":
    main()
