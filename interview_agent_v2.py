#!/usr/bin/env python3
"""
AI Mock Interviewer v2.0 (LangChain + File Memory + Custom Output CELL)
Домашнее задание №2 по курсу «LLM-агенты: от первого запроса до HARNESS».

Особенности реализации:
1. Архитектура на LangChain LCEL (LangChain Expression Language).
2. Долговременная память фактов (FileFactMemory): сохраняет профиль кандидата,
   пройденные темы, оценки и список оставшихся вопросов в memory.json.
3. Сохранение прогресса при перезапуске: агент не задает повторно уже пройденные вопросы.
4. Собственная CELL вывода (CustomInterviewCell): служебная информация о памяти
   и фактах выводится в отдельном блоке, а не смешивается с репликой интервьюера.
"""

import os
import sys
import json
import re
import select
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# LangChain Core & OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda


# ---------------------------------------------------------------------------
# 1. КОМПОНЕНТ ПАМЯТИ ФАКТОВ: FileFactMemory (Наследник Runnable)
# ---------------------------------------------------------------------------
class FileFactMemory(Runnable):
    """
    Память фактов из файла (не история диалога, а структурированные знания о пользователе).
    Читает и записывает memory.json, храня профиль кандидата и статус вопросов.
    """

    def __init__(self, filepath: str = "memory.json"):
        self.path = Path(filepath)
        self.data: Dict[str, Any] = self._load()

    def _default_data(self) -> Dict[str, Any]:
        return {
            "candidate": {
                "name": "Кандидат",
                "target_role": "Python Developer (Middle)",
                "skills": ["Python", "SQL", "Алгоритмы"]
            },
            "completed_questions": {},
            "pending_questions": [
                {
                    "id": "gil_threading",
                    "topic": "Python Core",
                    "question": "Как устроен GIL (Global Interpreter Lock) в CPython? В каких задачах он становится узким местом, а когда не влияет на производительность?"
                },
                {
                    "id": "sql_indexes_where_having",
                    "topic": "Базы данных / SQL",
                    "question": "В чем разница между WHERE и HAVING в SQL? Почему поиск по условию LIKE '%abc' не может эффективно использовать B-Tree индекс?"
                },
                {
                    "id": "bracket_balance_algo",
                    "topic": "Алгоритмы и структуры данных",
                    "question": "Опишите алгоритм проверки сбалансированности скобок в строке '()[]{}'. Каковы его временная и пространственная сложность?"
                },
                {
                    "id": "orm_n_plus_one",
                    "topic": "Архитектура и ORM",
                    "question": "Что такое проблема N+1 запросов при работе с ORM и какими методами (например, в Django ORM или SQLAlchemy) она решается?"
                }
            ],
            "session_meta": {
                "total_questions": 4,
                "last_updated": None
            }
        }

    def _load(self) -> Dict[str, Any]:
        """Загрузка данных из файла или инициализация значений по умолчанию."""
        if not self.path.exists():
            data = self._default_data()
            self._save(data)
            return data
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._default_data()

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Сохранение структуры памяти на диск."""
        if data is not None:
            self.data = data
        self.data["session_meta"]["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def invoke(self, input: Any = None, config=None, **kwargs) -> str:
        """
        Формирует компактный текстовый контекст фактов для вставки в {memory} промпта.
        """
        cand = self.data.get("candidate", {})
        name = cand.get("name", "Кандидат")
        role = cand.get("target_role", "Python Developer")
        completed = self.data.get("completed_questions", {})
        pending = self.data.get("pending_questions", [])

        lines = [
            f"Имя кандидата: {name}",
            f"Целевая позиция: {role}",
            f"Пройдено тем: {len(completed)} из {self.data.get('session_meta', {}).get('total_questions', 4)}"
        ]

        if completed:
            lines.append("Пройденные темы и оценки:")
            for q_id, info in completed.items():
                lines.append(f"  * [{info.get('topic', q_id)}]: {info.get('score', 0)}/10 (Отзыв: {info.get('feedback', '')})")
        else:
            lines.append("Пройденные темы: пока нет.")

        if pending:
            lines.append(f"Осталось тем для проверки: {len(pending)}")
            for p in pending:
                lines.append(f"  - id: {p['id']} | тема: {p['topic']}")
        else:
            lines.append("Все темы успешно пройдены!")

        return "\n".join(lines)

    def get_candidate_name(self) -> str:
        return self.data.get("candidate", {}).get("name", "Кандидат")

    def set_candidate_name(self, name: str):
        self.data["candidate"]["name"] = name.strip()
        self._save()

    def get_next_question(self) -> Optional[Dict[str, str]]:
        pending = self.data.get("pending_questions", [])
        if pending:
            return pending[0]
        return None

    def record_answer(self, q_id: str, score: int, feedback: str, user_answer: str) -> Dict[str, Any]:
        """Фиксирует ответ на вопрос, переносит тему из pending в completed и сохраняет файл."""
        pending = self.data.get("pending_questions", [])
        matched = None
        new_pending = []
        for q in pending:
            if q["id"] == q_id and matched is None:
                matched = q
            else:
                new_pending.append(q)

        topic = matched["topic"] if matched else q_id
        q_text = matched["question"] if matched else ""

        self.data["pending_questions"] = new_pending
        if "completed_questions" not in self.data:
            self.data["completed_questions"] = {}

        self.data["completed_questions"][q_id] = {
            "topic": topic,
            "question": q_text,
            "user_answer": user_answer[:300],
            "score": score,
            "feedback": feedback,
            "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save()
        return self.data["completed_questions"][q_id]

    def get_summary(self) -> str:
        completed = self.data.get("completed_questions", {})
        if not completed:
            return "Собеседование еще не начато."
        scores = [item["score"] for item in completed.values()]
        avg_score = round(sum(scores) / len(scores), 1)
        details = ", ".join(f"{item['topic']}: {item['score']}/10" for item in completed.values())
        return f"Средний балл: {avg_score}/10. Оценки по темам: {details}."


# ---------------------------------------------------------------------------
# 2. КАСТОМНАЯ CELL ВЫВОДА: CustomInterviewCell (Собственная CELL)
# ---------------------------------------------------------------------------
class CustomInterviewCell(Runnable):
    """
    Кастомный компонент-ячейка (CELL) для LangChain цепочки.
    Требование ДЗ №2: служебная информация о памяти и фактах должна
    выводиться через СОБСТВЕННУЮ CELL, не смешиваясь с репликой интервьюера.
    """

    def invoke(self, input: Any, config=None, **kwargs) -> Dict[str, Any]:
        raw_text = input if isinstance(input, str) else str(input)

        # Извлекаем оценку от 1 до 10
        score_match = re.search(r"ОЦЕНКА:\s*(\d+)", raw_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 8
        score = max(1, min(10, score))

        # Извлекаем комментарий
        feedback_match = re.search(r"ФИДБЕК:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
        feedback = feedback_match.group(1).strip() if feedback_match else "Ответ принят."

        # Реплика для кандидата (очищенная от служебных меток)
        dialogue = re.sub(r"ОЦЕНКА:\s*\d+", "", raw_text, flags=re.IGNORECASE)
        dialogue = re.sub(r"ФИДБЕК:\s*[^\n\r]+", "", dialogue, flags=re.IGNORECASE).strip()

        return {
            "dialogue": dialogue,
            "score": score,
            "feedback": feedback,
            "raw": raw_text
        }

    @staticmethod
    def render_service_cell(topic: str, q_id: str, score: int, feedback: str, remaining: int, memory_file: str):
        """Визуальный рендеринг отдельной ячейки со служебной информацией."""
        border = "─" * 68
        print(f"\n\033[1;34m┌{border}┐\033[0m")
        print(f"\033[1;34m│  💾 [СЛУЖЕБНАЯ CELL / ПАМЯТЬ АГЕНТА]\033[0m{' ' * 32}\033[1;34m│\033[0m")
        print(f"\033[1;34m├{border}┤\033[0m")
        print(f"\033[0;34m│  • Тема:           \033[1;37m{topic} ({q_id})\033[0m")
        print(f"\033[0;34m│  • Оценка в память: \033[1;33m{score}/10\033[0m")
        print(f"\033[0;34m│  • Фидбек модели:   \033[0;37m{feedback[:50]}\033[0m")
        print(f"\033[0;34m│  • Файл на диске:   \033[1;32m{memory_file} (успешно обновлен)\033[0m")
        print(f"\033[0;34m│  • Осталось тем:    \033[1;36m{remaining}\033[0m")
        print(f"\033[1;34m└{border}┘\033[0m\n")


# ---------------------------------------------------------------------------
# 3. ПОДДЕРЖКА МНОГОСТРОЧНОГО ВВОДА
# ---------------------------------------------------------------------------
def get_user_input() -> str:
    print("\n\033[1;32mКандидат > \033[0m", end="", flush=True)
    try:
        first_line = sys.stdin.readline()
    except (KeyboardInterrupt, EOFError):
        return "выход"

    if not first_line:
        return "выход"

    if first_line.strip() in [":code", ":paste", "```"]:
        print("\033[0;33m  📝 [Режим многострочного ввода. Вставьте код и введите :send или нажмите 2 раза Enter]:\033[0m")
        code_lines = []
        empty_count = 0
        while True:
            try:
                line = sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                break
            if not line:
                break
            if line.strip() in [":send", ":end", "```"]:
                break
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            code_lines.append(line)
        return "".join(code_lines).strip()

    # Автоперехват вставки из буфера обмена (50мс окно)
    lines = [first_line]
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            next_line = sys.stdin.readline()
            if not next_line:
                break
            lines.append(next_line)
        else:
            break

    return "".join(lines).strip()


# ---------------------------------------------------------------------------
# 4. ОСНОВНАЯ ПРОГРАММА И LANGCHAIN ЦЕПОЧКА
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  AI Mock Interviewer v2.0 (LangChain + File Memory + Custom CELL)")
    print("  Память фактов: memory.json (сохраняется между перезапусками)")
    print("  Команды:       'выход', 'exit', 'завершить'")
    print("  Ввод кода:     вставляйте напрямую или напишите ':code'")
    print("=" * 70)

    # 1. Инициализация памяти из файла
    memory = FileFactMemory("memory.json")
    candidate_name = memory.get_candidate_name()

    # 2. Инициализация локальной LLM через LangChain
    base_url = os.getenv("LLAMA_API_BASE", "http://127.0.0.1:8080/v1")
    llm = ChatOpenAI(
        model="local-model",
        base_url=base_url,
        api_key="not-needed",
        temperature=0.1
    )

    # 3. Системный промпт с переменной {memory}
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — Senior Tech Lead и профессиональный технический интервьюер.\n"
         "Твоя задача — оценить ответ кандидата на заданный технический вопрос.\n\n"
         "АКТУАЛЬНАЯ ПАМЯТЬ О КАНДИДАТЕ (из файла):\n"
         "-----------------------------------------\n"
         "{memory}\n"
         "-----------------------------------------\n\n"
         "ПРАВИЛА ОТВЕТА:\n"
         "1. В начале ответа укажи две служебные строки строго в формате:\n"
         "   ОЦЕНКА: <число от 1 до 10>\n"
         "   ФИДБЕК: <краткий комментарий до 15 слов, что было хорошо, а что упущено>\n"
         "2. Затем с новой строки дай развернутый, конструктивный комментарий кандидату."),
        ("human",
         "Вопрос интервью: {question_text}\n"
         "Ответ кандидата:\n{candidate_answer}")
    ])

    # 4. Сборка LCEL-цепочки
    custom_cell = CustomInterviewCell()
    chain = (
        {
            "memory": memory,
            "question_text": lambda x: x["question_text"],
            "candidate_answer": lambda x: x["candidate_answer"]
        }
        | eval_prompt
        | llm
        | StrOutputParser()
        | custom_cell
    )

    # 5. Проверка состояния памяти при старте
    completed_count = len(memory.data.get("completed_questions", {}))
    pending_count = len(memory.data.get("pending_questions", []))

    if completed_count > 0 and pending_count > 0:
        print(f"\n\033[1;36mИнтервьюер >\033[0m С возвращением, {candidate_name}!")
        print(f"  В памяти уже сохранены результаты по {completed_count} темам.")
        print("  Продолжаем собеседование с оставшихся тем.\n")
    elif completed_count > 0 and pending_count == 0:
        print(f"\n\033[1;32mИнтервьюер >\033[0m Здравствуйте, {candidate_name}!")
        print(f"  Вы уже полностью прошли все темы собеседования!")
        print(f"  🏆 {memory.get_summary()}")
        print("  Если хотите пройти заново, удалите файл memory.json.")
        return
    else:
        print(f"\n\033[1;36mИнтервьюер >\033[0m Здравствуйте, {candidate_name}!")
        print("  Рад приветствовать вас на техническом собеседовании. Начнем по порядку.\n")

    # 6. Основной цикл опроса по оставшимся вопросам
    while True:
        next_q = memory.get_next_question()
        if not next_q:
            print("\n" + "=" * 70)
            print("  🎉 ВСЕ ВОПРОСЫ УСПЕШНО ПРОЙДЕНЫ!")
            print(f"  🏆 {memory.get_summary()}")
            print("  Итоговый протокол сохранен в memory.json.")
            print("=" * 70)
            break

        q_id = next_q["id"]
        topic = next_q["topic"]
        q_text = next_q["question"]

        print(f"\033[1;37m❓ [Тема: {topic}]:\033[0m {q_text}")

        # Считываем ответ кандидата
        user_answer = get_user_input()

        if not user_answer:
            continue

        if user_answer.lower() in ["выход", "exit", "quit", "завершить"]:
            print(f"\n\033[1;33m[Собеседование приостановлено]\033[0m")
            print(f"Прогресс зафиксирован в memory.json. При следующем запуске продолжим с темы: {topic}.")
            break

        # Прогон через LangChain LCEL цепочку
        try:
            print("\n\033[0;33m[Анализ ответа моделью через LangChain...]\033[0m")
            result = chain.invoke({
                "question_text": q_text,
                "candidate_answer": user_answer
            })
        except Exception as e:
            print(f"\033[1;31m[Ошибка вызова LLM через LangChain]: {e}\033[0m")
            print("Убедитесь, что llama-server запущен на порту 8080!")
            break

        # 1. Сохраняем факт в memory.json
        score = result["score"]
        feedback = result["feedback"]
        memory.record_answer(q_id=q_id, score=score, feedback=feedback, user_answer=user_answer)

        # 2. Отрисовываем служебную информацию через собственную CELL (требование ДЗ)
        remaining = len(memory.data.get("pending_questions", []))
        CustomInterviewCell.render_service_cell(
            topic=topic,
            q_id=q_id,
            score=score,
            feedback=feedback,
            remaining=remaining,
            memory_file="memory.json"
        )

        # 3. Выводим содержательный комментарий интервьюера кандидату
        print(f"\033[1;36mИнтервьюер >\033[0m {result['dialogue']}\n")


if __name__ == "__main__":
    main()
