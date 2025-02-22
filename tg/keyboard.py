from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton

QUIT_BUTTON = "Quit conversation"
CHOOSE_MODEL_BUTTON = "Choose solution approach"

reply_keyboard = [
    [
        QUIT_BUTTON, 
        CHOOSE_MODEL_BUTTON
    ]
]

reply_keyboard_markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=False)

def get_response_inline_keyboard(buttons):
    # Create a list of InlineKeyboardButton objects, 2 in a row
    keyboard = []
    for i in range(0, len(buttons), 2):
        row = []
        for j in range(2):
            if i + j < len(buttons):
                row.append(InlineKeyboardButton(buttons[i + j], callback_data=buttons[i + j]))
        keyboard.append(row)

    return InlineKeyboardMarkup(keyboard)