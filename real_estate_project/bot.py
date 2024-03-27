import telebot
import numpy as np
from real_estate import predict_price
Token = "7043837569:AAGeQmA5hSgXq44fk_pwh6g9B-eXxvc40D8"

bot = telebot.TeleBot(Token)

@bot.message_handler(['start'])
def start(message):
    bot.reply_to(message, '''Hello! I am a Real Estate Property Value Predictor Bot. I will take values of various parameters from you, and would predict the price of the property.\n\nThese are the parameters used for predicting the prices : 
    1. CRIM        per capita crime rate by town
    2. ZN          proportion of residential land zoned for lots over 
                   25,000 sq.ft.
    3. INDUS       proportion of non-retail business acres per town
    4. CHAS        Charles River dummy variable (= 1 if tract bounds 
                   river; 0 otherwise)
    5. NOX         nitric oxides concentration (parts per 10 million)
    6. RM          average number of rooms per dwelling
    7. AGE         proportion of owner-occupied units built prior to 1940
    8. DIS         weighted distances to five Boston employment centres
    9. RAD         index of accessibility to radial highways
    10. TAX        full-value property-tax rate per $10,000
    11. PTRATIO    pupil-teacher ratio by town
    12. B          1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                   by town
    13. LSTAT      percentage of lower status of the population
                 
    Please provide 13 comma separated values in the same order as mentioned in the list.''' )

@bot.message_handler()
def predict(message):
    try:
        user_input = [list(map(float, message.text.split(',')))]
        count = 0
        for listElem in user_input:
            count += len(listElem)
        if count != 13:
            bot.reply_to(message, "Please provide exactly 13 values.")
            return
        
        else:
            features = np.array(user_input)
            predicted_price = predict_price(features)
            bot.reply_to(message, f"Predicted Price = ${predicted_price*1000}")

    except Exception as e:
        bot.reply_to(message, "Error processing your request")

bot.polling()
