#!/usr/bin/env python3
"""
AI Mock Interviewer v3.0
Лабораторная работа №3 (Занятия 04 и 05).

Реализованные механизмы из Занятий 04 и 05:
1. Комбинированная память:
   - Оконная память (ConversationBufferWindowMemory, k=2 последних обмена на базе InMemoryChatMessageHistory).
   - Суммаризирующая память (ConversationSummaryMemory — сжатие истории диалога с помощью LLM).
   - Файловая память фактов и состояния (memory_v3.json — уровень кандидата, пройденные темы, банк вопросов).
2. Кастомный компонент с разбором входных параметров (CustomParameterHandlerCell):
   - Наследник Runnable, анализирующий входные параметры, результаты классификации и вызывающий @tool инструменты.
3. Условное ветвление:
   - Схема классификации ответа через Pydantic (Sentiment & Intent) + PydanticOutputParser.
   - Маршрутизация через RunnableBranch:
     * Ветка негатива/усталости (эмпатия + смягчение нагрузки).
     * Ветка запроса подсказки (вызов @tool get_topic_hint без штрафа по баллам).
     * Ветки оценки по уровню знаний из памяти (Junior / Middle / Senior).
4. Инструменты @tool и bind_tools:
   - Набор инструментов @tool: fetch_question_by_level, get_topic_hint, check_python_syntax, record_score_and_adapt_level.
   - Диспетчер выполнения tool_calls (run_tools_as_messages) и интеграция в LCEL-пайплайн.
"""

import os
import sys
import ast
import json
import re
import select
import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

MEMORY_FILE = "memory_v3.json"


# 1. КОМПОНЕНТЫ ПАМЯТИ
class InterviewBufferWindowMemory:
    """
    Оконная память (аналог ConversationBufferWindowMemory из Занятия 04).
    Хранит последние k обменов репликами поверх InMemoryChatMessageHistory.
    """

    def __init__(self, k: int = 2, initial_pairs: Optional[List[Dict[str, str]]] = None):
        self.k = k
        self.chat_memory = InMemoryChatMessageHistory()
        if initial_pairs:
            for pair in initial_pairs[-self.k:]:
                self.chat_memory.add_user_message(pair.get("input", ""))
                self.chat_memory.add_ai_message(pair.get("output", ""))

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_text = str(inputs.get("input", ""))
        ai_text = str(outputs.get("output", ""))
        self.chat_memory.add_user_message(user_text)
        self.chat_memory.add_ai_message(ai_text)
        # Обрезаем историю до последних k пар (2 * k сообщений)
        max_msgs = self.k * 2
        if len(self.chat_memory.messages) > max_msgs:
            self.chat_memory.messages = self.chat_memory.messages[-max_msgs:]

    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        lines = []
        for msg in self.chat_memory.messages:
            role = "Кандидат" if isinstance(msg, HumanMessage) else "Интервьюер"
            lines.append(f"{role}: {msg.content}")
        return {"history": "\n".join(lines) if lines else "История окна пока пуста."}

    def to_serializable(self) -> List[Dict[str, str]]:
        pairs = []
        msgs = self.chat_memory.messages
        for i in range(0, len(msgs), 2):
            u = msgs[i].content if i < len(msgs) else ""
            a = msgs[i + 1].content if (i + 1) < len(msgs) else ""
            pairs.append({"input": u, "output": a})
        return pairs[-self.k:]


class InterviewSummaryMemory:
    """
    Суммаризирующая память.
    Сжимает историю собеседования в краткую сводку с помощью LLM.
    """

    def __init__(self, llm: ChatOpenAI, initial_summary: str = ""):
        self.llm = llm
        self.summary = initial_summary or "Собеседование только началось."
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Ты — модуль суммаризации памяти (ConversationSummaryMemory). "
             "Обнови краткую сводку технического собеседования (не более 3 предложений на русском языке), "
             "объединив текущее резюме и новый обмен репликами. Сохраняй ключевые факты об ответах и уровне кандидата."),
            ("human",
             "Текущая сводка:\n{current_summary}\n\n"
             "Новый обмен репликами:\nКандидат: {user_input}\nИнтервьюер: {ai_output}\n\n"
             "Обновленная краткая сводка:")
        ])
        self.chain = self.summary_prompt | self.llm | StrOutputParser()

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        user_input = str(inputs.get("input", ""))
        ai_output = str(outputs.get("output", ""))
        try:
            updated = self.chain.invoke({
                "current_summary": self.summary,
                "user_input": user_input[:400],
                "ai_output": ai_output[:400],
            }).strip()
            if updated:
                self.summary = updated
        except Exception:
            # Резервное обновление при недоступности сервера
            self.summary = f"{self.summary} | Последний ответ: {user_input[:80]}..."

    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {"summary": self.summary}


class HybridFileMemory(Runnable):
    """
    Объединенное файловое хранилище (memory_v3.json), связывающее:
    1. Факты о кандидате и динамический уровень знаний (junior / middle / senior).
    2. Оконную память (InterviewBufferWindowMemory, k=2).
    3. Суммаризирующую память (InterviewSummaryMemory).
    """

    def __init__(self, filepath: str = MEMORY_FILE, llm: Optional[ChatOpenAI] = None):
        self.path = Path(filepath)
        self.data: Dict[str, Any] = self._load()
        layers = self.data.get("memory_layers", {})
        self.window_memory = InterviewBufferWindowMemory(
            k=layers.get("window_k", 2),
            initial_pairs=layers.get("window_buffer", [])
        )
        self.summary_memory = InterviewSummaryMemory(
            llm=llm,
            initial_summary=layers.get("dialogue_summary", "")
        ) if llm else None

    def _default_data(self) -> Dict[str, Any]:
        return {
            "candidate": {
                "name": "Виктор",
                "target_role": "Python Developer",
                "current_level": "middle",
                "skills": ["Python", "SQL", "Алгоритмы", "FastAPI", "LangChain"]
            },
            "memory_layers": {
                "window_k": 2,
                "window_buffer": [],
                "dialogue_summary": "Собеседование только начинается. Стартовый уровень кандидата: middle."
            },
            "completed_questions": {},
            "question_bank": {
                "junior": [
                    {
                        "id": "jun_list_vs_tuple",
                        "topic": "Python База",
                        "level": "junior",
                        "question": "В чем ключевые отличия списков (list) от кортежей (tuple) в Python? Может ли кортеж быть ключом словаря dict?",
                        "hint": "Подумайте про изменяемость (mutability), хешируемость (hashable) и объем занимаемой памяти."
                    },
                    {
                        "id": "jun_sql_joins",
                        "topic": "SQL База",
                        "level": "junior",
                        "question": "Чем отличается INNER JOIN от LEFT JOIN в SQL? Приведите пример, когда LEFT JOIN вернет NULL.",
                        "hint": "Вспомните, что происходит со строками левой таблицы, для которых не нашлось совпадений в правой."
                    }
                ],
                "middle": [
                    {
                        "id": "mid_gil_threading",
                        "topic": "Python Core",
                        "level": "middle",
                        "question": "Как устроен GIL (Global Interpreter Lock) в CPython? В каких задачах он становится узким местом, а когда не мешает?",
                        "hint": "Разделите задачи на две категории: вычисления на процессоре (CPU-bound) и ожидание ввода-вывода (I/O-bound)."
                    },
                    {
                        "id": "mid_sql_indexes",
                        "topic": "Базы данных / SQL",
                        "level": "middle",
                        "question": "В чем разница между WHERE и HAVING в SQL? Почему условие LIKE '%abc' не может эффективно использовать B-Tree индекс?",
                        "hint": "WHERE фильтрует строки до группировки GROUP BY, а B-Tree индекс упорядочивает строки слева направо по префиксу."
                    },
                    {
                        "id": "mid_decorators_code",
                        "topic": "Python Практика (Код)",
                        "level": "middle",
                        "question": "Напишите небольшой декоратор или функцию на Python для замера времени выполнения функции (или опишите ее структуру с сохранением метаданных через functools.wraps).",
                        "hint": "Используйте time.perf_counter(), вложенную функцию wrapper(*args, **kwargs) и @functools.wraps(func)."
                    }
                ],
                "senior": [
                    {
                        "id": "sen_asyncio_internals",
                        "topic": "Архитектура / Asyncio",
                        "level": "senior",
                        "question": "Как внутри работает Event Loop в asyncio? Что произойдет, если внутри async-корутины вызвать блокирующий CPU-код или синхронный requests.get(), и как это правильно изолировать?",
                        "hint": "Подумайте о кооперативной многозадачности, системных вызовах epoll/kqueue и пуле потоков loop.run_in_executor / asyncio.to_thread."
                    },
                    {
                        "id": "sen_distributed_tx",
                        "topic": "Highload & Микросервисы",
                        "level": "senior",
                        "question": "Как обеспечить консистентность данных между двумя микросервисами и брокером сообщений (Kafka/RabbitMQ), если распределенные 2PC-транзакции недоступны?",
                        "hint": "Вспомните паттерны Transactional Outbox, Saga (хореография/оркестрация) и идемпотентность консьюмеров."
                    }
                ]
            },
            "session_meta": {
                "last_updated": None,
                "total_turns": 0
            }
        }

    def _load(self) -> Dict[str, Any]:
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
        if data is not None:
            self.data = data
        if hasattr(self, "window_memory") and self.window_memory:
            self.data["memory_layers"]["window_buffer"] = self.window_memory.to_serializable()
        if hasattr(self, "summary_memory") and self.summary_memory:
            self.data["memory_layers"]["dialogue_summary"] = self.summary_memory.summary
        self.data["session_meta"]["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def reload_from_disk(self) -> None:
        self.data = self._load()

    def get_level(self) -> str:
        return self.data.get("candidate", {}).get("current_level", "middle")

    def get_candidate_name(self) -> str:
        return self.data.get("candidate", {}).get("name", "Кандидат")

    def update_interaction(self, user_input: str, ai_output: str) -> None:
        """Обновляет и буферное окно (k=2), и LLM-суммаризацию, и сохраняет файл."""
        self.reload_from_disk()
        self.window_memory.save_context({"input": user_input}, {"output": ai_output})
        if self.summary_memory:
            self.summary_memory.save_context({"input": user_input}, {"output": ai_output})
        self.data["session_meta"]["total_turns"] = self.data["session_meta"].get("total_turns", 0) + 1
        self._save()

    def invoke(self, input: Any = None, config=None, **kwargs) -> str:
        """Возвращает полный срез всех 3 уровней памяти для подстановки в промпт."""
        self.reload_from_disk()
        cand = self.data.get("candidate", {})
        completed = self.data.get("completed_questions", {})
        win_text = self.window_memory.load_memory_variables()["history"]
        sum_text = (
            self.summary_memory.load_memory_variables()["summary"]
            if self.summary_memory
            else self.data.get("memory_layers", {}).get("dialogue_summary", "")
        )

        lines = [
            f"Имя кандидата: {cand.get('name', 'Виктор')} | Роль: {cand.get('target_role', 'Python Developer')}",
            f"Текущий адаптивный уровень знаний: {cand.get('current_level', 'middle').upper()}",
            f"Пройдено вопросов: {len(completed)}",
            f"[ConversationSummaryMemory (Сводка диалога)]: {sum_text}",
            f"[ConversationBufferWindowMemory (Последние k={self.window_memory.k} реплик)]:\n{win_text}",
        ]
        return "\n".join(lines)


@tool
def fetch_question_by_level(level: str = "middle") -> str:
    """
    Выбирает следующий непройденный вопрос из банка вопросов (memory_v3.json)
    в соответствии с текущим уровнем знаний кандидата ('junior', 'middle' или 'senior').
    Если на текущем уровне вопросы закончились, берет доступный вопрос с соседнего уровня.
    """
    path = Path(MEMORY_FILE)
    if not path.exists():
        return json.dumps({"error": "Файл памяти не найден"}, ensure_ascii=False)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    completed_ids = set(data.get("completed_questions", {}).keys())
    bank = data.get("question_bank", {})

    # Порядок поиска вопросов с учетом текущего уровня
    search_order = [level.lower()] + [lvl for lvl in ["middle", "senior", "junior"] if lvl != level.lower()]

    for lvl in search_order:
        for q in bank.get(lvl, []):
            if q["id"] not in completed_ids:
                return json.dumps(q, ensure_ascii=False)

    return json.dumps({"status": "ALL_COMPLETED"}, ensure_ascii=False)


@tool
def get_topic_hint(question_id: str) -> str:
    """
    Возвращает методическую подсказку по идентификатору вопроса (question_id),
    когда кандидат просит наводящую мысль или затрудняется с ответом.
    """
    path = Path(MEMORY_FILE)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        bank = data.get("question_bank", {})
        for lvl_questions in bank.values():
            for q in lvl_questions:
                if q["id"] == question_id:
                    return f"Подсказка по теме '{q['topic']}': {q.get('hint', 'Подумайте о базовых принципах работы механизма.')}"
    return "Подсказка: разбейте задачу на базовые составляющие и опишите основной сценарий использования."


@tool
def check_python_syntax(code: str) -> str:
    """
    Проверяет синтаксическую корректность фрагмента кода Python через AST-парсер
    и определяет использованные языковые конструкции (функции, декораторы, исключения).
    """
    cleaned = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.MULTILINE)
    try:
        tree = ast.parse(cleaned)
        features = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features.append(f"функция '{node.name}'")
            elif isinstance(node, ast.ClassDef):
                features.append(f"класс '{node.name}'")
            elif isinstance(node, ast.Try):
                features.append("блок try/except")
            elif isinstance(node, ast.Return):
                features.append("оператор return")
        feat_str = ", ".join(dict.fromkeys(features)) if features else "базовые выражения"
        return f"AST-проверка пройдена успешно: синтаксис Python корректен. Обнаружены конструкции: {feat_str}."
    except SyntaxError as err:
        return f"AST-проверка выявила синтаксическую ошибку в строке {err.lineno}: {err.msg}."


@tool
def record_score_and_adapt_level(question_id: str, score: int, feedback: str, user_answer: str) -> str:
    """
    Сохраняет оценку за вопрос в memory_v3.json и динамически адаптирует уровень
    кандидата (junior <-> middle <-> senior) на основе набранного балла:
    - score >= 8: повышение уровня (junior -> middle -> senior)
    - score <= 4: понижение уровня (senior -> middle -> junior)
    - 5 <= score <= 7: уровень остается прежним.
    """
    path = Path(MEMORY_FILE)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    old_level = data["candidate"].get("current_level", "middle").lower()
    levels = ["junior", "middle", "senior"]
    idx = levels.index(old_level) if old_level in levels else 1

    if score >= 8 and idx < 2:
        new_level = levels[idx + 1]
    elif score <= 4 and idx > 0:
        new_level = levels[idx - 1]
    else:
        new_level = old_level

    data["candidate"]["current_level"] = new_level

    # Находим метаданные вопроса в банке
    topic, q_text, q_lvl = question_id, "", old_level
    for lvl, q_list in data.get("question_bank", {}).items():
        for q in q_list:
            if q["id"] == question_id:
                topic = q["topic"]
                q_text = q["question"]
                q_lvl = q["level"]
                break

    data.setdefault("completed_questions", {})[question_id] = {
        "topic": topic,
        "question_level": q_lvl,
        "question": q_text,
        "user_answer": user_answer[:300],
        "score": score,
        "feedback": feedback,
        "level_transition": f"{old_level} -> {new_level}",
        "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    data["session_meta"]["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json.dumps({
        "question_id": question_id,
        "score": score,
        "old_level": old_level,
        "new_level": new_level,
        "total_completed": len(data["completed_questions"])
    }, ensure_ascii=False)


# Справочник всех инструментов для динамического диспетчера
INTERVIEW_TOOLS = [
    fetch_question_by_level,
    get_topic_hint,
    check_python_syntax,
    record_score_and_adapt_level,
]
TOOLS_BY_NAME = {t.name: t for t in INTERVIEW_TOOLS}


def run_tools_as_messages(ai_msg: AIMessage) -> List[Any]:
    """
    Универсальный диспетчер tool_calls.
    Выполняет все запрошенные моделью инструменты и упаковывает результаты в ToolMessage.
    """
    msgs: List[Any] = [ai_msg]
    for tc in getattr(ai_msg, "tool_calls", []):
        tool_obj = TOOLS_BY_NAME.get(tc["name"])
        if tool_obj:
            result = tool_obj.invoke(tc["args"])
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return msgs


# 2. СХЕМА КЛАССИФИКАЦИИ И КАСТОМНЫЙ КОМПОНЕНТ РАЗБОРА ПАРАМЕТРОВ
class CandidateInputSchema(BaseModel):
    intent: Literal["technical_answer", "ask_hint", "negative_or_tired"] = Field(
        description=(
            "Класс реплики пользователя: "
            "'technical_answer' — попытка ответить на вопрос или код; "
            "'ask_hint' — просьба дать подсказку, наводку или сказать 'не знаю'; "
            "'negative_or_tired' — жалоба на усталость, раздражение, негатив или 'слишком сложно'."
        )
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Тональность сообщения кандидата: 'positive', 'neutral' или 'negative'"
    )
    has_code: bool = Field(
        description="True, если ответ содержит фрагмент кода на Python (def, import, return, class и т.п.)"
    )
    reason: str = Field(
        description="Краткое пояснение выбора классификации в одном предложении"
    )


class CustomParameterInspectorCell(Runnable):
    """
    'custom компонент с разбором входных параметров — с действием tools или элементом памяти'.

    Разбирает входные параметры пайплайна:
    1. Запускает классификацию ответа кандидата (LLM + PydanticOutputParser с надежным fallback).
    2. В зависимости от параметров автоматически вызывает нужные @tool:
       - если обнаружен код (has_code=True) -> вызывает @tool check_python_syntax;
       - если запрошена подсказка (intent='ask_hint') -> вызывает @tool get_topic_hint.
    3. Извлекает контекст из HybridFileMemory (Window + Summary + Facts).
    """

    def __init__(self, llm: ChatOpenAI, memory: HybridFileMemory):
        self.llm = llm
        self.memory = memory
        self.parser = PydanticOutputParser(pydantic_object=CandidateInputSchema)
        self.classifier_prompt = PromptTemplate(
            template=(
                "Проанализируй ответ кандидата на техническом собеседовании.\n"
                "Определи намерение (intent), тональность (sentiment) и наличие кода Python (has_code).\n\n"
                "Вопрос интервьюера: {question_text}\n"
                "Реплика кандидата: {candidate_answer}\n\n"
                "{format_instruction}"
            ),
            input_variables=["question_text", "candidate_answer"],
            partial_variables={"format_instruction": self.parser.get_format_instructions()},
        )

    def _heuristic_classify(self, text: str) -> CandidateInputSchema:
        """Страховка для компактных 3B-моделей при сбое JSON-схемы."""
        low = text.lower().strip()
        neg_words = [
            "устал", "сложно", "бесит", "надоело", "ужас", "плохо",
            "не понимаю ничего", "тяжело", "хватит мучить", "разочарован"
        ]
        hint_words = [
            "подсказк", "подскажи", "наводк", "помоги", "затрудняюсь",
            "не помню", "не знаю", "hint", "help", "напомни"
        ]
        code_markers = ["def ", "return ", "import ", "class ", "try:", "except", "@wraps", "async def "]

        has_code = any(m in text for m in code_markers)
        if any(w in low for w in neg_words):
            return CandidateInputSchema(
                intent="negative_or_tired",
                sentiment="negative",
                has_code=has_code,
                reason="Обнаружены маркеры усталости или негативной эмоциональной окраски."
            )
        if any(w in low for w in hint_words) and len(low) < 120:
            return CandidateInputSchema(
                intent="ask_hint",
                sentiment="neutral",
                has_code=False,
                reason="Кандидат явно запросил методическую подсказку по вопросу."
            )
        return CandidateInputSchema(
            intent="technical_answer",
            sentiment="positive" if len(text) > 120 else "neutral",
            has_code=has_code,
            reason="Кандидат предоставил содержательный технический ответ на вопрос."
        )

    def invoke(self, inputs: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        q_id = inputs["question_id"]
        q_text = inputs["question_text"]
        q_topic = inputs.get("question_topic", "Python")
        answer = inputs["candidate_answer"]

        # 1. Сначала проверяем явные триггеры, затем вызываем LLM-классификатор с PydanticOutputParser
        heur = self._heuristic_classify(answer)
        if heur.intent in ("ask_hint", "negative_or_tired"):
            classification = heur
        else:
            try:
                raw_cls = (self.classifier_prompt | self.llm | StrOutputParser()).invoke({
                    "question_text": q_text,
                    "candidate_answer": answer
                })
                classification = self.parser.parse(raw_cls)
                # Если в ответе явно есть код, гарантируем флаг has_code
                if heur.has_code:
                    classification.has_code = True
            except Exception:
                classification = heur

        # 2. Вызов инструментов (@tool) по результатам разбора входных параметров
        called_tools: List[str] = []
        tool_observations: List[str] = []

        if classification.has_code:
            syntax_report = check_python_syntax.invoke({"code": answer})
            called_tools.append("check_python_syntax")
            tool_observations.append(syntax_report)

        if classification.intent == "ask_hint":
            hint_text = get_topic_hint.invoke({"question_id": q_id})
            called_tools.append("get_topic_hint")
            tool_observations.append(hint_text)

        # 3. Загружаем состояние памяти (уровень кандидата + буфер окна + сводка)
        current_level = self.memory.get_level()
        memory_context = self.memory.invoke()

        return {
            **inputs,
            "question_topic": q_topic,
            "classification": classification,
            "intent": classification.intent,
            "sentiment": classification.sentiment,
            "has_code": classification.has_code,
            "current_level": current_level,
            "memory_context": memory_context,
            "called_tools": called_tools,
            "tool_observations": "\n".join(tool_observations) if tool_observations else "Инструменты пре-обработки не требовались.",
        }


# 3. УСЛОВНЫЕ ВЕТКИ (RunnableBranch) И КАСТОМНАЯ СЕРВИСНАЯ ЯЧЕЙКА ВЫВОДА
def build_branching_pipeline(llm: ChatOpenAI, memory: HybridFileMemory) -> Runnable:
    """
    Собирает полный условный конвейер LCEL с использованием RunnableBranch (Занятие 04).
    Ветвление зависит от:
    1) Класса ответа и тональности кандидата (negative_or_tired / ask_hint / technical_answer).
    2) Уровня знаний кандидата в памяти (junior / middle / senior).
    """
    inspector = CustomParameterInspectorCell(llm=llm, memory=memory)

    # Ветка 1: Негатив или усталость кандидата (эмпатия + предложение упростить)
    empathy_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — чуткий и поддерживающий Tech Lead. Кандидат устал, расстроен или считает вопрос слишком сложным.\n"
         "Контекст памяти:\n{memory_context}\n\n"
         "Правила ответа:\n"
         "1. Первой строкой напиши: РЕЖИМ: EMPATHY_SUPPORT\n"
         "2. Поддержи кандидата, снизь стресс и дай краткую наводку по вопросу, чтобы ему было проще ответить."),
        ("human",
         "Текущий вопрос: {question_text}\n"
         "Реплика кандидата: {candidate_answer}")
    ])

    # Ветка 2: Запрос подсказки (работает вместе с @tool get_topic_hint)
    hint_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — наставник на техническом интервью. Кандидат попросил подсказку.\n"
         "Инструмент @tool get_topic_hint вернул следующую информацию:\n{tool_observations}\n\n"
         "Правила ответа:\n"
         "1. Первой строкой напиши: РЕЖИМ: HINT_PROVIDED\n"
         "2. Мягко и понятно сформулируй подсказку на основе данных инструмента, не раскрывая полный ответ целиком, "
         "и предложи кандидату попробовать ответить еще раз."),
        ("human",
         "Вопрос: {question_text}\n"
         "Запрос кандидата: {candidate_answer}")
    ])

    # Ветка 3: Строгая оценка уровня SENIOR
    senior_eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — Principal System Architect. Уровень кандидата в памяти: SENIOR.\n"
         "Оценивай ответ строго: обращай внимание на архитектурные компромиссы, внутреннее устройство и производительность.\n"
         "Контекст памяти:\n{memory_context}\n"
         "Результат проверки кода (если был): {tool_observations}\n\n"
         "ПРАВИЛА ОТВЕТА (строго соблюдай первые две строки):\n"
         "ОЦЕНКА: <число от 1 до 10>\n"
         "ФИДБЕК: <краткое резюме оценки в 1 предложение>\n"
         "Затем дай глубокий профессиональный комментарий для Senior-разработчика."),
        ("human",
         "Вопрос (Senior): {question_text}\n"
         "Ответ кандидата:\n{candidate_answer}")
    ])

    # Ветка 4: Обучающая оценка уровня JUNIOR
    junior_eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — доброжелательный ментор для Junior Python-разработчика. Уровень кандидата в памяти: JUNIOR.\n"
         "Оценивай ответ лояльно, хвали за верные базовые мысли и простыми словами объясняй упущенные детали.\n"
         "Контекст памяти:\n{memory_context}\n"
         "Результат проверки кода (если был): {tool_observations}\n\n"
         "ПРАВИЛА ОТВЕТА (строго соблюдай первые две строки):\n"
         "ОЦЕНКА: <число от 1 до 10>\n"
         "ФИДБЕК: <краткое резюме оценки в 1 предложение>\n"
         "Затем дай понятный, ободряющий разбор ответа."),
        ("human",
         "Вопрос (Junior): {question_text}\n"
         "Ответ кандидата:\n{candidate_answer}")
    ])

    # Ветка 5 (По умолчанию / Default): Сбалансированная оценка уровня MIDDLE
    middle_eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — Senior Tech Lead. Уровень кандидата в памяти: MIDDLE.\n"
         "Оцени точность ответа, понимание практики и наличие примеров.\n"
         "Контекст памяти:\n{memory_context}\n"
         "Результат проверки кода (если был): {tool_observations}\n\n"
         "ПРАВИЛА ОТВЕТА (строго соблюдай первые две строки):\n"
         "ОЦЕНКА: <число от 1 до 10>\n"
         "ФИДБЕК: <краткое резюме оценки в 1 предложение>\n"
         "Затем дай конструктивный комментарий кандидату."),
        ("human",
         "Вопрос (Middle): {question_text}\n"
         "Ответ кандидата:\n{candidate_answer}")
    ])

    # Развилка RunnableBranch (обязательно со списком пар условие-цепочка и веткой по умолчанию)
    branch_router = RunnableBranch(
        (
            lambda x: x["intent"] == "negative_or_tired" or x["sentiment"] == "negative",
            RunnablePassthrough.assign(
                active_branch=lambda _: "empathy_negative_branch",
                llm_output=empathy_prompt | llm | StrOutputParser()
            ),
        ),
        (
            lambda x: x["intent"] == "ask_hint",
            RunnablePassthrough.assign(
                active_branch=lambda _: "hint_mentor_branch",
                llm_output=hint_prompt | llm | StrOutputParser()
            ),
        ),
        (
            lambda x: x["current_level"] == "senior",
            RunnablePassthrough.assign(
                active_branch=lambda _: "senior_strict_eval_branch",
                llm_output=senior_eval_prompt | llm | StrOutputParser()
            ),
        ),
        (
            lambda x: x["current_level"] == "junior",
            RunnablePassthrough.assign(
                active_branch=lambda _: "junior_friendly_eval_branch",
                llm_output=junior_eval_prompt | llm | StrOutputParser()
            ),
        ),
        # Ветка по умолчанию (обязательна в RunnableBranch)
        RunnablePassthrough.assign(
            active_branch=lambda _: "middle_default_eval_branch",
            llm_output=middle_eval_prompt | llm | StrOutputParser()
        ),
    )

    return inspector | branch_router | CustomServiceOutputCell(memory=memory)


class CustomServiceOutputCell(Runnable):
    """
    Финальная сервисная CELL пайплайна:
    - Извлекает оценку и фидбек (если сработала оценочная ветка).
    - Вызывает инструмент @tool record_score_and_adapt_level для сохранения в memory_v3.json.
    - Обновляет ConversationBufferWindowMemory и ConversationSummaryMemory.
    - Рендерит изолированный сервисный блок в консоль отдельно от реплики интервьюера.
    """

    def __init__(self, memory: HybridFileMemory):
        self.memory = memory

    def invoke(self, state: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        raw_text = state.get("llm_output", "")
        branch = state.get("active_branch", "middle_default_eval_branch")
        cls_obj: CandidateInputSchema = state["classification"]
        called_tools: List[str] = list(state.get("called_tools", []))
        q_id = state["question_id"]
        q_topic = state.get("question_topic", q_id)
        user_answer = state["candidate_answer"]

        question_completed = False
        score: Optional[int] = None
        feedback = ""
        old_level = state.get("current_level", "middle")
        new_level = old_level

        if branch in ("empathy_negative_branch", "hint_mentor_branch"):
            # Вопрос остается активным, штраф не начисляется
            dialogue = re.sub(r"РЕЖИМ:\s*[A-Z_]+\s*", "", raw_text).strip()
            feedback = (
                "Выдана методическая подсказка (вопрос остается активным)"
                if branch == "hint_mentor_branch"
                else "Поддержка кандидата при усталости/негативе"
            )
            # Если кандидат жалуется на сложность на уровне senior/middle — адаптируем уровень мягче
            if branch == "empathy_negative_branch" and old_level == "senior":
                self.memory.data["candidate"]["current_level"] = "middle"
                self.memory._save()
                new_level = "middle"
        else:
            question_completed = True
            score_match = re.search(r"ОЦЕНКА:\s*(\d+)", raw_text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 8
            score = max(1, min(10, score))

            fb_match = re.search(r"ФИДБЕК:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
            feedback = fb_match.group(1).strip() if fb_match else "Ответ оценен."

            dialogue = re.sub(r"ОЦЕНКА:\s*\d+", "", raw_text, flags=re.IGNORECASE)
            dialogue = re.sub(r"ФИДБЕК:\s*[^\n\r]+", "", dialogue, flags=re.IGNORECASE).strip()

            # Вызываем @tool для сохранения результата и адаптации уровня в memory_v3.json
            tool_res_raw = record_score_and_adapt_level.invoke({
                "question_id": q_id,
                "score": score,
                "feedback": feedback,
                "user_answer": user_answer
            })
            called_tools.append("record_score_and_adapt_level")
            try:
                tool_res = json.loads(tool_res_raw)
                old_level = tool_res.get("old_level", old_level)
                new_level = tool_res.get("new_level", old_level)
            except Exception:
                pass

        # Обновляем оконную память (k=2) и LLM-суммаризацию
        self.memory.update_interaction(user_input=user_answer, ai_output=dialogue)

        # Отрисовываем изолированную сервисную CELL
        self.render_cell(
            topic=q_topic,
            q_id=q_id,
            cls_obj=cls_obj,
            branch=branch,
            called_tools=called_tools,
            tool_obs=state.get("tool_observations", ""),
            score=score,
            feedback=feedback,
            old_level=old_level,
            new_level=new_level,
        )

        return {
            "dialogue": dialogue,
            "question_completed": question_completed,
            "score": score,
            "branch": branch,
            "new_level": new_level,
            "called_tools": called_tools,
        }

    def render_cell(
        self,
        topic: str,
        q_id: str,
        cls_obj: CandidateInputSchema,
        branch: str,
        called_tools: List[str],
        tool_obs: str,
        score: Optional[int],
        feedback: str,
        old_level: str,
        new_level: str,
    ) -> None:
        border = "─" * 74
        tools_str = ", ".join(called_tools) if called_tools else "нет"
        score_str = f"{score}/10" if score is not None else "без оценки (промежуточный шаг)"
        lvl_str = (
            f"{old_level.upper()} -> {new_level.upper()} (адаптирован!)"
            if old_level != new_level
            else f"{new_level.upper()} (без изменений)"
        )
        summary_preview = (
            self.memory.summary_memory.summary[:85] + "..."
            if self.memory.summary_memory and len(self.memory.summary_memory.summary) > 85
            else (self.memory.summary_memory.summary if self.memory.summary_memory else "")
        )
        win_count = len(self.memory.window_memory.to_serializable())

        print(f"\n\033[1;34m┌{border}┐\033[0m")
        print(f"\033[1;34m│  ⚙️  [CUSTOM SERVICE CELL: ПАРАМЕТРЫ · ВЕТВЛЕНИЕ · TOOLS · ПАМЯТЬ]{' ' * 8}│\033[0m")
        print(f"\033[1;34m├{border}┤\033[0m")
        print(f"\033[0;34m│  • Вопрос / Тема:      \033[1;37m{topic} ({q_id})\033[0m")
        print(
            f"\033[0;34m│  • Классификация:      \033[1;36mintent={cls_obj.intent} | "
            f"sentiment={cls_obj.sentiment} | has_code={cls_obj.has_code}\033[0m"
        )
        print(f"\033[0;34m│  • Ветка RunnableBranch:\033[1;35m {branch}\033[0m")
        print(f"\033[0;34m│  • Вызванные @tool:    \033[1;32m{tools_str}\033[0m")
        if tool_obs and "не требовались" not in tool_obs:
            print(f"\033[0;34m│  • Результат @tool:    \033[0;37m{tool_obs[:68]}\033[0m")
        print(f"\033[0;34m│  • Оценка за шаг:      \033[1;33m{score_str}\033[0m ({feedback[:42]})")
        print(f"\033[0;34m│  • Уровень знаний:     \033[1;33m{lvl_str}\033[0m")
        print(f"\033[0;34m│  • WindowMemory (k=2): \033[0;37mхранится {win_count} последних пар(ы) реплик\033[0m")
        print(f"\033[0;34m│  • SummaryMemory:      \033[0;37m{summary_preview}\033[0m")
        print(f"\033[0;34m│  • Файл памяти:        \033[1;32m{MEMORY_FILE} синхронизирован\033[0m")
        print(f"\033[1;34m└{border}┘\033[0m\n")


# 4. ВВОД ПОЛЬЗОВАТЕЛЯ И ЗАПУСК (ИНТЕРАКТИВНЫЙ РЕЖИМ + РЕЖИМ --demo)
def get_user_input() -> str:
    print("\n\033[1;32mКандидат > \033[0m", end="", flush=True)
    try:
        first_line = sys.stdin.readline()
    except (KeyboardInterrupt, EOFError):
        return "выход"

    if not first_line:
        return "выход"

    if first_line.strip() in [":code", ":paste", "```"]:
        print("\033[0;33m  📝 [Многострочный режим. Вставьте код и введите :send или дважды нажмите Enter]:\033[0m")
        code_lines = []
        empty_count = 0
        while True:
            try:
                line = sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                break
            if not line or line.strip() in [":send", ":end", "```"]:
                break
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            code_lines.append(line)
        return "".join(code_lines).strip()

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


def run_demo_scenario(pipeline: Runnable, memory: HybridFileMemory) -> None:
    """
    Автоматический демонстрационный прогон всех возможностей Лабораторной работы №3:
    1) Запрос подсказки -> срабатывание ветки 'hint_mentor_branch' и @tool get_topic_hint.
    2) Сильный ответ с кодом Python -> @tool check_python_syntax + оценка 9/10 + повышение уровня middle -> senior.
    3) Эмоциональная реплика об усталости -> ветка 'empathy_negative_branch' и адаптация нагрузки.
    """
    print("\n\033[1;35m=== ЗАПУСК АВТОМАТИЧЕСКОЙ ДЕМОНСТРАЦИИ (ЛАБОРАТОРНАЯ РАБОТА №3) ===\033[0m\n")

    demo_steps = [
        {
            "desc": "ШАГ 1: Кандидат просит подсказку (проверка ветки ask_hint + @tool get_topic_hint)",
            "answer": "Честно говоря, немного забыл детали про GIL и потоки, дайте, пожалуйста, небольшую подсказку."
        },
        {
            "desc": "ШАГ 2: Кандидат дает развернутый ответ с примером кода Python (проверка @tool check_python_syntax + повышение уровня middle -> senior)",
            "answer": (
                "GIL (Global Interpreter Lock) — это глобальная блокировка интерпретатора CPython. "
                "В CPU-bound задачах она мешает параллелизму потоков, поэтому там нужен multiprocessing. "
                "А в I/O-bound задачах GIL отпускается при ожидании сети или диска.\n"
                "def worker(n: int) -> int:\n"
                "    try:\n"
                "        return sum(i * i for i in range(n))\n"
                "    except Exception:\n"
                "        return 0"
            )
        },
        {
            "desc": "ШАГ 3: На следующем (Senior) вопросе кандидат жалуется на усталость (проверка ветки empathy_negative_branch)",
            "answer": "Ох, я уже очень устал сегодня, слишком сложный архитектурный вопрос, тяжело соображать..."
        }
    ]

    for step in demo_steps:
        lvl = memory.get_level()
        q_raw = fetch_question_by_level.invoke({"level": lvl})
        q_data = json.loads(q_raw)
        if q_data.get("status") == "ALL_COMPLETED":
            break

        print(f"\033[1;33m>>> {step['desc']}\033[0m")
        print(f"\033[1;37m❓ [Уровень: {q_data['level'].upper()} | Тема: {q_data['topic']}]:\033[0m {q_data['question']}")
        print(f"\033[1;32mКандидат >\033[0m {step['answer']}")

        result = pipeline.invoke({
            "question_id": q_data["id"],
            "question_topic": q_data["topic"],
            "question_text": q_data["question"],
            "candidate_answer": step["answer"],
        })
        print(f"\033[1;36mИнтервьюер >\033[0m {result['dialogue']}\n")
        print("-" * 76)


def main():
    print("=" * 76)
    print("  AI Mock Interviewer v3.0 (LangChain: Window+Summary Memory, Branching, @tool)")
    print(f"  Файл памяти:  {MEMORY_FILE} (профиль, адаптивный уровень, окно k=2, сводка)")
    print("  Команды:      'выход', 'exit' | Для подсказки напишите: 'подскажи'")
    print("  Демо-режим:   python3 interview_agent_v3.py --demo")
    print("=" * 76)

    base_url = os.getenv("LLAMA_API_BASE", "http://127.0.0.1:8080/v1")
    llm = ChatOpenAI(
        model="local-model",
        base_url=base_url,
        api_key="not-needed",
        temperature=0.1,
    )

    memory = HybridFileMemory(filepath=MEMORY_FILE, llm=llm)
    pipeline = build_branching_pipeline(llm=llm, memory=memory)

    if "--demo" in sys.argv:
        run_demo_scenario(pipeline, memory)
        return

    cand_name = memory.get_candidate_name()
    completed_cnt = len(memory.data.get("completed_questions", {}))
    curr_lvl = memory.get_level().upper()

    print(f"\n\033[1;36mИнтервьюер >\033[0m Приветствую, {cand_name}!")
    print(f"  Текущий адаптивный уровень: \033[1;33m{curr_lvl}\033[0m | Пройдено вопросов: {completed_cnt}.")
    print(f"  Сводка памяти: {memory.summary_memory.summary}\n")

    current_q: Optional[Dict[str, Any]] = None

    while True:
        if current_q is None:
            lvl = memory.get_level()
            q_raw = fetch_question_by_level.invoke({"level": lvl})
            q_data = json.loads(q_raw)
            if q_data.get("status") == "ALL_COMPLETED":
                print("\n" + "=" * 76)
                print("  🎉 ВСЕ ВОПРОСЫ БАНКА УСПЕШНО ПРОЙДЕНЫ!")
                print(f"  Итоговый уровень кандидата: {memory.get_level().upper()}")
                print(f"  Финальная сводка (SummaryMemory): {memory.summary_memory.summary}")
                print("=" * 76)
                break
            current_q = q_data

        print(
            f"\033[1;37m❓ [Уровень: {current_q['level'].upper()} | "
            f"Тема: {current_q['topic']}]:\033[0m {current_q['question']}"
        )

        user_answer = get_user_input()
        if not user_answer:
            continue

        if user_answer.lower() in ["выход", "exit", "quit", "завершить"]:
            print(f"\n\033[1;33m[Сессия сохранена в {MEMORY_FILE}]\033[0m")
            print(f"Текущий уровень: {memory.get_level().upper()}. До встречи!")
            break

        try:
            print("\n\033[0;33m[Классификация ответа, выбор ветки RunnableBranch и вызов @tool...]\033[0m")
            result = pipeline.invoke({
                "question_id": current_q["id"],
                "question_topic": current_q["topic"],
                "question_text": current_q["question"],
                "candidate_answer": user_answer,
            })
        except Exception as e:
            print(f"\033[1;31m[Ошибка выполнения цепочки]: {e}\033[0m")
            print("Проверьте, что запущен llama-server на порту 8080.")
            break

        print(f"\033[1;36mИнтервьюер >\033[0m {result['dialogue']}\n")

        # Если вопрос был оценен — переходим к следующему (с учетом нового уровня!).
        # Если была выдана подсказка или эмоциональная поддержка — оставляем вопрос для ответа.
        if result["question_completed"]:
            current_q = None


if __name__ == "__main__":
    main()
