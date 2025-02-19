#!/bin/bash

LOG_FILE="logs/initial_file_dump.log"
echo "🔍 Starting file import: $(date)" > "$LOG_FILE"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

project_prefix="${PROJECT_NAME}"

EXCLUDE_DIRS=("node_modules" "public" "build" "dist" ".git" "venv" "__pycache__")

is_frontend_file() {
    case "$1" in
        *.js|*.jsx|*.css|*.html|*.ts|*.tsx|*.json) return 0 ;; 
        *) return 1 ;;
    esac
}

process_file() {
    local file_path="$1"
    local relative_path="${file_path#*/frontend/}"

    for dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            return
        fi
    done

    if ! is_frontend_file "$relative_path"; then
        return
    fi

    printf "⏳ Processing: $relative_path\n" | tee -a "$LOG_FILE"

    content=$(cat "$file_path" | sed "s/'/''/g")

    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "
        INSERT INTO ${project_prefix}_files (path, content, file_category) 
        VALUES ('$relative_path', '$content', 'frontend')
        ON DUPLICATE KEY UPDATE content=VALUES(content), file_category=VALUES(file_category);
    " 2>>"$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo "❌ Error inserting file: $relative_path" >> "$LOG_FILE"
        return
    fi

    file_id=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" -N -B "$DB_NAME" -e "SELECT id FROM ${project_prefix}_files WHERE path='$relative_path' LIMIT 1;" 2>>"$LOG_FILE")

    if [[ -z "$file_id" ]]; then
        echo "⚠️ Warning: File ID not found for $relative_path" >> "$LOG_FILE"
        return
    fi

    git --git-dir="$PROJECT_ROOT/.git" --work-tree="$PROJECT_ROOT" blame --line-porcelain "$file_path" | while read -r line; do
        if [[ $line =~ ^([0-9a-f]{40}) ]]; then
            commit_hash="${BASH_REMATCH[1]}"
        fi

        if [[ $line =~ ^author\ (.*) ]]; then
            author="${BASH_REMATCH[1]}"
        fi

        if [[ $line =~ ^committer-time\ ([0-9]+) ]]; then
            commit_date=$(date -r "${BASH_REMATCH[1]}" "+%Y-%m-%d %H:%M:%S")
        fi

        if [[ ! $line =~ ^(author|committer|summary|previous|filename|boundary|[0-9a-f]{40}) ]]; then
            content="$line" 
            ((line_number++))

            printf "Blame >> Author: $author, Commit: $commit_hash, Date: $commit_date, Line: $line_number, Content: $content\n" | tee -a "$LOG_FILE"

            mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "
                INSERT INTO ${project_prefix}_files_blame (file_id, path, line_number, author, commit_hash, commit_date, content)
                VALUES ('$file_id', '$relative_path', '$line_number', '$author', '$commit_hash', '$commit_date', '$(echo "$content" | sed "s/'/''/g")')
                ON DUPLICATE KEY UPDATE 
                    author=VALUES(author), 
                    commit_hash=VALUES(commit_hash), 
                    commit_date=VALUES(commit_date), 
                    content=VALUES(content);
            " 2>>"$LOG_FILE"

            if [ $? -ne 0 ]; then
                echo "❌ Error inserting blame info for file: $relative_path (line $line_number)" >> "$LOG_FILE"
            fi
        fi
    done

    echo "✅ Successfully imported: $relative_path" | tee -a "$LOG_FILE"
}

export -f process_file is_frontend_file

find "$PROJECT_ROOT" -type d \( -name "dist" -o -name "node_modules" -o -name "public" -o -name "build" -o -name ".git" -o -name "venv" -o -name "__pycache__" \) -prune -o -type f -print | xargs -P 4 -n 1 bash -c 'process_file "$@"' _

echo "🎉 Import complete: $(date)" | tee -a "$LOG_FILE"
echo "📄 Log saved at: $LOG_FILE"