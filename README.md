# Reply AI Challenge - Fraud Detection System

Multi-agent AI system designed to detect fraudulent financial transactions using GPS, behavioral patterns, and communication analysis (LLM).

## Project Structure

- **`software/`**: Source code containing detection agents (GPS, Behavior, Comms) and decision logic.
- **`dataset/`**: Directory for input data (place the world data folders here).
- **`output/`**: Results directory containing detailed English reports and transaction IDs.
- **`.venv/`**: Python virtual environment with all required dependencies.

## Setup

The virtual environment is already configured. To activate it:

```bash
source .venv/bin/activate
```

## How to Run

1. **Add Data**: Copy the folder containing the data you want to analyze (e.g., `Brave New World - train`) into the `dataset/` directory.
   
2. **Execute**: Run the program from the `software` directory, pointing to the dataset path:

   ```bash
   cd software
   python main.py --world "../dataset/YOUR_DATASET_FOLDER_NAME"
   ```

3. **Check Results**: After execution, check the `output/` directory for:
   - `output_[dataset_name].txt`: A detailed English report with reasoning for each flagged fraud.
   - `fraud_ids_[dataset_name].txt`: A simple list of flagged transaction IDs.
   - `patterns_[dataset_name].json`: Learned patterns to improve detection in subsequent levels.

## Requirements

- Python 3.10+
- LangChain / OpenAI (OpenRouter)
- Langfuse (for tracing)
- Pandas & python-dotenv
