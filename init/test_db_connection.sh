#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "⏳ Проверка подключения к MySQL серверу: $DB_HOST:$DB_PORT"

ERROR_MSG=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" -e "SHOW DATABASES;" 2>&1)

if [ $? -eq 0 ]; then
    echo "✅ Успешное подключение к MySQL!"
    echo "📋 Список таблиц в базе данных $DB_NAME:"
    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASS" -D "$DB_NAME" -e "SHOW TABLES;"
else
    echo "❌ Ошибка подключения к MySQL: $ERROR_MSG"
fi