TSLA Streamlit App

This Streamlit app visualizes TSLA stock data with a candlestick chart, a Gemini API chatbot, and an animated replay of the chart, similar to a TradingView backtest. The app is deployed on Streamlit Cloud at https://tsla-app-app-kkgnwh59w4waulii6gqf7j.streamlit.app/.
Features

Chart Tab: Displays a candlestick chart of TSLA stock data with support/resistance bands and direction markers ("SHORT", "LONG", "None"). Includes a range menu to select date ranges (e.g., August 2022 or full dataset).
Chatbot Tab: A Gemini API-powered chatbot that answers questions about the TSLA data (e.g., "How many days in 2023 was TSLA bullish?").
Note: The chatbot is currently limited by the Gemini API free tier quota (error 429) until the quota resets at 12:30 PM IST on May 24, 2025.


Animation Tab: An animated replay of the candlestick chart, bands, and markers, with a range menu for selecting date ranges (e.g., 2022, 2023, full range).

Project Structure

app.py: Main Streamlit app script.
data/tsla.csv: TSLA stock dataset.
requirements.txt: Python dependencies for the app.

Setup and Installation

Clone the repository:git clone https://github.com/KartavyaChauhan/tsla-streamlit-app.git
cd tsla-streamlit-app


Install dependencies:pip install -r requirements.txt


Set the Gemini API key as an environment variable:export GEMINI_API_KEY="your_api_key_here"

On Windows (PowerShell):$Env:GEMINI_API_KEY = "your_api_key_here"


Run the app locally:streamlit run app.py



Deployment

The app is deployed on Streamlit Cloud. To deploy your own instance:
Push the code to a GitHub repository.
Link the repository to Streamlit Cloud and deploy.
Add GEMINI_API_KEY to Streamlit Cloud secrets.

Notes

The Gemini API quota limitation affects the chatbot tab until the reset time. The chart and animation tabs are fully functional.
For large date ranges in the animation tab, performance may vary due to the number of frames.
