# StackMind Tutorial

Welcome to the StackMind tutorial! This guide will walk you through the core capabilities of the StackMind problem-solving engine.

Unlike standard chatbots, StackMind is designed to break down problems, reason through them step-by-step, and verify its own solutions.

## Prerequisites

Ensure you have installed the dependencies and set up your `.env` file as described in the `README.md`.

---

## 1. Using the Command Line Interface (CLI)

The CLI is the easiest way to interact with StackMind locally. Start it by running:

```bash
python main.py --mode cli
```

You will see the interactive prompt `StackMind> `.

### Text-Based Problem Solving

Try asking a complex reasoning question that requires multi-step logic.

**Example 1: Math and Logic**
```
StackMind> If a train travels 60 mph for 2 hours, then 80 mph for 1 hour, what is its average speed?
```
*Notice how StackMind breaks this into finding the total distance and total time before calculating the final answer.*

**Example 2: Code Debugging**
```
StackMind> /domain code
StackMind> Debug this Python function:
def fibonacci(n):
    if n <= 0: return []
    elif n == 1: return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
    return result[n] # Bug here!
```
*StackMind will detect the bug (returning `result[n]` instead of `result`), explain it, and provide the fixed code.*

### Multi-Modal Processing

StackMind supports processing different types of input files directly from the CLI.

**Image Processing:**
```
StackMind> /image path/to/diagram.png Explain what this architecture diagram represents.
```

**Voice Processing:**
```
StackMind> /voice path/to/recording.mp3 Please summarize the key points from this audio clip.
```

**Code Analysis:**
```
StackMind> /code path/to/script.py Optimize this code for better performance.
```

### CLI Commands

Type `/help` to see all available commands:
- `/history` - View your solution history
- `/clear` - Clear the current session
- `/domain [name]` - Set a specific domain hint (e.g., math, code, science)
- `/exit` or `quit` - Exit the CLI

---

## 2. Using the REST API

For integrating StackMind into other applications, use the REST API. Start the server:

```bash
python main.py --mode api
```

The API runs on `http://127.0.0.1:8000` by default. You can view the interactive documentation at `http://127.0.0.1:8000/docs`.

### Example: Solving a Text Problem via API

You can interact with the API using `curl` or any HTTP client (like Postman or Python's `requests` library).

```bash
curl -X POST "http://127.0.0.1:8000/solve/text" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "What are the primary differences between deductive and inductive reasoning?",
           "domain": "general"
         }'
```

**Response Structure:**
The API will return a structured JSON response containing:
- `solution`: The final answer.
- `reasoning_steps`: The chain-of-thought breakdown.
- `domain`: The domain solver used.
- `confidence_score`: The self-verification score (if applicable).

### Example: Analyzing Code via API

```bash
curl -X POST "http://127.0.0.1:8000/solve/code" \
     -H "Content-Type: application/json" \
     -d '{
           "code": "def greet(name): print(\"Hello \" + name)",
           "language": "python",
           "domain": "code"
         }'
```

---

## 3. Understanding the Output

When you use StackMind, pay attention to the structured output:

1.  **The Plan:** StackMind first creates a plan (e.g., "Step 1: Parse input. Step 2: Analyze logic. Step 3: Formulate answer.")
2.  **Reasoning Chain:** It executes the plan step-by-step.
3.  **Verification:** It reviews its own work to catch hallucinations or logic errors.
4.  **Final Solution:** The synthesized result.

This transparent process is what sets StackMind apart from standard "black box" language models.