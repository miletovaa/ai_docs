#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/init.log"
echo "🚀 Initialization started at $(date)" > "$LOG_FILE"

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "🟢 Activating virtual environment..." | tee -a "$LOG_FILE"
    source "$PROJECT_ROOT/venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT"

echo "🛠️ Running database migrations..." | tee -a "$LOG_FILE"
python3 -m init.migrations >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: migrations.py failed! Check logs/init.log for details." | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Migrations completed successfully." | tee -a "$LOG_FILE"

echo "📂 Running initial file dump..." | tee -a "$LOG_FILE"
bash "$PROJECT_ROOT/init/initial_file_dump.sh" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: initial_file_dump.sh failed! Check logs/init.log for details." | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Initial file dump completed successfully." | tee -a "$LOG_FILE"

echo "🎉 All tasks completed successfully at $(date)" | tee -a "$LOG_FILE"