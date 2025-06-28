# StackMind Tutorial

## Getting Started

1. **Prerequisites**
   - Python 3.8 or higher installed
   - Git (optional for cloning the repository)
   - API keys for your chosen LLM provider (Mistral or OpenAI)

2. **Installation**
   ```bash
   git clone https://github.com/your-repo/StackMind.git
   cd StackMind
   pip install -r requirements.txt
   ```

3. **Configuration**
   - Create a `.env` file in the project root with your API key:
     ```
     MISTRAL_API_KEY=your_key_here
     # or
     OPENAI_API_KEY=your_key_here
     ```
   - Edit `config/settings.py` if you need to customize default settings

## Running the Application

### Command Line Interface
```bash
python main.py
```

### API Mode
```bash
python main.py --api
```

## Basic Usage

1. **Text Input**
   - Type your problem directly at the prompt
   - Example: "Solve the equation 2x + 5 = 15"

2. **Image Processing**
   - Use the `image` command followed by file path
   - Example: `image math_problem.png`

3. **Code Analysis**
   - Use the `code` command followed by file path
   - Example: `code example.py`

4. **Voice Input**
   - Use the `voice` command followed by audio file path
   - Example: `voice question.wav`

## Advanced Features

- **Solution Refinement**: Use `feedback "your feedback"` to improve solutions
- **History**: View past solutions with `history` command
- **Export**: Save solutions with `save filename.txt`

## Troubleshooting

- If you get API key errors, verify your `.env` file
- For debug information, run with `--debug` flag
- Check `problem_solver.log` for detailed logs

## Examples

```
$ python main.py
> What is the capital of France?
[AI responds with answer and reasoning]
> feedback "Please provide more historical context"
[AI provides refined answer with history]
> history
[Shows list of previous solutions]
```