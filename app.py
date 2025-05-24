import pandas as pd
from time import sleep
from lightweight_charts import Chart
import ast
import streamlit as st
import google.generativeai as genai

# Initialize Gemini API with enhanced error handling
gemini_available = False
model = None
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_available = True
except FileNotFoundError:
    st.error("Secrets file not found at C:\\Users\\karta\\OneDrive\\Desktop\\tsla_dashboard\\.streamlit\\secrets.toml. Please create it with GOOGLE_API_KEY.")
except KeyError:
    st.error("GOOGLE_API_KEY not found in secrets.toml. Please add it in the format: GOOGLE_API_KEY = \"your-key\".")
except Exception as e:
    st.error(f"Error parsing secrets.toml: {str(e)}. Ensure it contains valid TOML syntax, e.g., GOOGLE_API_KEY = \"your-key\".")
if not gemini_available:
    st.warning("Chatbot functionality disabled due to API key issues. Chart will still display.")

# Load TSLA data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tsla_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={'timestamp': 'time'})
        
        if 'Support' in df.columns:
            df['Support'] = df['Support'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            df['Support'] = [[] for _ in range(len(df))]
            df['Support'] = df['close'].apply(lambda x: [x - 20, x - 10, x])
        
        if 'Resistance' in df.columns:
            df['Resistance'] = df['Resistance'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            df['Resistance'] = [[] for _ in range(len(df))]
            df['Resistance'] = df['close'].apply(lambda x: [x + 10, x + 20, x + 30])
        
        if 'direction' not in df.columns:
            df['direction'] = None
        return df
    except FileNotFoundError:
        st.error("tsla_data.csv not found")
        return pd.DataFrame()

# Chart functions
def get_bar_data(symbol, timeframe):
    if symbol != 'TSLA':
        st.warning(f'No data for "{symbol}"')
        return pd.DataFrame()
    df = load_data()
    if df.empty:
        return df
    return df

def on_search(chart, searched_string):
    new_data = get_bar_data(searched_string, chart.topbar['timeframe'].value)
    if new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.set(new_data[['time', 'open', 'high', 'low', 'close', 'volume']])

def on_timeframe_selection(chart):
    new_data = get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data.empty:
        return
    chart.set(new_data[['time', 'open', 'high', 'low', 'close', 'volume']], True)

def on_horizontal_line_move(chart, line):
    st.write(f'Horizontal line moved to: {line.price}')

def render_chart():
    # Initialize chart only if not already done
    if 'chart_initialized' not in st.session_state:
        st.session_state.chart_initialized = False
        st.session_state.chart = None

    if not st.session_state.chart_initialized:
        chart = Chart(toolbox=True)
        chart.events.search += on_search
        chart.topbar.textbox('symbol', 'TSLA')
        chart.topbar.switcher('timeframe', ('1min', '5min', '30min'), default='5min', func=on_timeframe_selection)

        chart.layout(background_color='#090008', text_color='#FFFFFF', font_size=16, font_family='Helvetica')
        chart.candle_style(up_color='#00ff55', down_color='#ed4807', border_up_color='#FFFFFF', border_down_color='#FFFFFF', wick_up_color='#FFFFFF', wick_down_color='#FFFFFF')
        chart.volume_config(up_color='#00ff55', down_color='#ed4807')
        chart.watermark('1D', color='rgba(180, 180, 240, 0.7)')
        chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted', horz_color='#FFFFFF', horz_style='dotted')
        chart.legend(visible=True, font_size=14)
        chart.horizontal_line(200, func=on_horizontal_line_move)

        df = get_bar_data('TSLA', '5min')
        if df.empty:
            st.error("No data to display.")
            return

        initial_rows = 20
        chart.set(df[['time', 'open', 'high', 'low', 'close', 'volume']].iloc[:initial_rows])
        
        support_lower = chart.create_line('Support Lower', color='#00FF55', width=3, style='dashed')
        support_upper = chart.create_line('Support Upper', color='#00FF55', width=3, style='dashed')
        resistance_lower = chart.create_line('Resistance Lower', color='#ED4807', width=3, style='dashed')
        resistance_upper = chart.create_line('Resistance Upper', color='#ED4807', width=3, style='dashed')

        try:
            chart.show(block=False)  # Open chart in a separate window
            st.session_state.chart_initialized = True
            st.session_state.chart = chart
            st.session_state.support_lower = support_lower
            st.session_state.support_upper = support_upper
            st.session_state.resistance_lower = resistance_lower
            st.session_state.resistance_upper = resistance_upper
            st.session_state.df = df
            st.session_state.initial_rows = initial_rows
            st.session_state.last_close = df.iloc[initial_rows - 1]['close']
            st.session_state.support_lower_points = []
            st.session_state.support_upper_points = []
            st.session_state.resistance_lower_points = []
            st.session_state.resistance_upper_points = []
        except Exception as e:
            st.error(f"Error initializing chart: {str(e)}")
            return

    # Use stored chart and data
    chart = st.session_state.chart
    df = st.session_state.df
    support_lower = st.session_state.support_lower
    support_upper = st.session_state.support_upper
    resistance_lower = st.session_state.resistance_lower
    resistance_upper = st.session_state.resistance_upper
    initial_rows = st.session_state.initial_rows
    last_close = st.session_state.last_close
    support_lower_points = st.session_state.support_lower_points
    support_upper_points = st.session_state.support_upper_points
    resistance_lower_points = st.session_state.resistance_lower_points
    resistance_upper_points = st.session_state.resistance_upper_points

    # Continue animation only if not already completed
    if 'current_row' not in st.session_state:
        st.session_state.current_row = initial_rows

    for i in range(st.session_state.current_row, len(df)):
        row = df.iloc[i]
        chart.update(row[['time', 'open', 'high', 'low', 'close', 'volume']])

        if row['direction'] == 'LONG':
            chart.marker(time=row['time'], position='below', shape='arrow_up', color='#00ff55', text='LONG')
        elif row['direction'] == 'SHORT':
            chart.marker(time=row['time'], position='above', shape='arrow_down', color='#ed4807', text='SHORT')
        elif row['direction'] is None:
            chart.marker(time=row['time'], position='above', shape='square', color="#FFFF00", text='Neutral', size=20)

        if row['Support'] and isinstance(row['Support'], list) and len(row['Support']) > 0 and all(isinstance(x, (int, float)) for x in row['Support']):
            support_lower_points.append({'time': row['time'], 'Support Lower': min(row['Support'])})
            support_upper_points.append({'time': row['time'], 'Support Upper': max(row['Support'])})
        if row['Resistance'] and isinstance(row['Resistance'], list) and len(row['Resistance']) > 0 and all(isinstance(x, (int, float)) for x in row['Resistance']):
            resistance_lower_points.append({'time': row['time'], 'Resistance Lower': min(row['Resistance'])})
            resistance_upper_points.append({'time': row['time'], 'Resistance Upper': max(row['Resistance'])})

        if support_lower_points:
            support_lower.set(pd.DataFrame(support_lower_points))
            support_upper.set(pd.DataFrame(support_upper_points))
        if resistance_lower_points:
            resistance_lower.set(pd.DataFrame(resistance_lower_points))
            resistance_upper.set(pd.DataFrame(resistance_upper_points))

        if row['close'] > 20 and last_close < 20:
            chart.marker(text='The price crossed $20!')
        last_close = row['close']
        st.session_state.last_close = last_close
        st.session_state.current_row = i + 1
        sleep(0.1)

# Chatbot function (unchanged)
def chatbot_interface():
    st.header("TSLA Data Chatbot")
    if not gemini_available:
        st.warning("Chatbot is disabled due to API key issues.")
        return

    st.write("Ask questions about the TSLA stock data (e.g., 'How many days in 2023 was TSLA bullish?')")

    df = load_data()
    if df.empty:
        st.error("No data available for the chatbot.")
        return

    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    bullish_days = df[df['close'] > df['open']].groupby('year').size().to_string()
    signal_counts = df.groupby('year')['direction'].value_counts().unstack(fill_value=0).to_string()
    neutral_signals = df[df['direction'].isna() | (df['direction'] == 'None')].groupby(['year', 'month']).size().unstack(fill_value=0).to_string()
    price_stats = df.groupby('year').agg({
        'high': 'max',
        'low': 'min',
        'volume': 'mean'
    }).to_string()
    sample_rows = df.groupby('year').apply(lambda x: x.sample(1)).to_string()
    context = f"""You are a financial analyst assistant. You have access to TSLA stock data from 2022 to May 2025 with columns: {', '.join(df.columns)}. 
    Summary statistics:
    - Bullish days (close > open) per year:\n{bullish_days}
    - Signal counts per year (LONG, SHORT, None):\n{signal_counts}
    - Neutral signals (direction is None or empty) per year and month:\n{neutral_signals}
    - Price and volume stats per year:\n{price_stats}
    Sample data (one row per year):\n{sample_rows}
    Answer questions based on this data. Provide concise and accurate answers."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about TSLA data")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            prompt = f"{context}\n\nQuestion: {user_input}"
            response = model.generate_content(prompt)
            answer = response.text
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    st.subheader("Example Questions")
    example_questions = [
        "Which month in 2022 had the most Neutral signals?",
        "How many days in 2023 was TSLA bullish (close > open)?",
        "What was the highest price of TSLA in 2024?",
        "How many LONG signals occurred in the data?",
        "What is the average volume on days with a SHORT signal in 2025?"
    ]
    for q in example_questions:
        st.write(f"- {q}")

# Main app
def main():
    st.title("TSLA Stock Dashboard")
    tab1, tab2 = st.tabs(["Candlestick Chart", "Chatbot"])

    with tab1:
        st.header("TSLA Candlestick Chart")
        st.write("The chart will open in a separate window.")
        render_chart()

    with tab2:
        chatbot_interface()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows multiprocessing
    main()
                                 #python -m streamlit run app.py