import pandas as pd
from time import sleep
import json
import streamlit as st
import streamlit.components.v1 as components
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
        
        # Parse Support and Resistance columns
        if 'Support' in df.columns:
            df['Support'] = df['Support'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        else:
            df['Support'] = [[] for _ in range(len(df))]
            df['Support'] = df['close'].apply(lambda x: [x - 20, x - 10, x])
        
        if 'Resistance' in df.columns:
            df['Resistance'] = df['Resistance'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
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

def render_chart():
    # Load data
    df = get_bar_data('TSLA', '5min')
    if df.empty:
        st.error("No data to display.")
        return

    # Prepare candlestick data
    candlestick_data = []
    for _, row in df.iterrows():
        timestamp = int(row['time'].timestamp())
        candlestick_data.append({
            "time": timestamp,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close'])
        })

    # Prepare volume data
    volume_data = []
    for _, row in df.iterrows():
        timestamp = int(row['time'].timestamp())
        volume_data.append({
            "time": timestamp,
            "value": float(row['volume']),
            "color": "#00ff55" if row['close'] >= row['open'] else "#ed4807"
        })

    # Prepare support and resistance lines
    support_lower_data = []
    support_upper_data = []
    resistance_lower_data = []
    resistance_upper_data = []
    for _, row in df.iterrows():
        timestamp = int(row['time'].timestamp())
        if row['Support'] and isinstance(row['Support'], list) and len(row['Support']) > 0 and all(isinstance(x, (int, float)) for x in row['Support']):
            support_lower_data.append({"time": timestamp, "value": min(row['Support'])})
            support_upper_data.append({"time": timestamp, "value": max(row['Support'])})
        if row['Resistance'] and isinstance(row['Resistance'], list) and len(row['Resistance']) > 0 and all(isinstance(x, (int, float)) for x in row['Resistance']):
            resistance_lower_data.append({"time": timestamp, "value": min(row['Resistance'])})
            resistance_upper_data.append({"time": timestamp, "value": max(row['Resistance'])})

    # Prepare markers with explicit price levels
    markers = []
    for _, row in df.iterrows():
        timestamp = int(row['time'].timestamp())
        current_close = float(row['close'])
        current_high = float(row['high'])
        current_low = float(row['low'])
        candle_height = current_high - current_low
        buffer = candle_height * 0.2  # 20% of candle height for spacing

        marker = {"time": timestamp}
        direction = row['direction']

        if direction == 'LONG':
            marker.update({
                "position": "belowBar",
                "shape": "arrowUp",
                "color": "#00ff55",  # Green
                "text": "LONG",
                "price": current_low - buffer  # Below the candlestick
            })
        elif direction == 'SHORT':
            marker.update({
                "position": "aboveBar",
                "shape": "arrowDown",
                "color": "#ed4807",  # Red
                "text": "SHORT",
                "price": current_high + buffer  # Above the candlestick
            })
        else:  # Neutral
            marker.update({
                "position": "inBar",
                "shape": "circle",
                "color": "#FFFF00",  # Yellow
                "text": "Neutral",
                "price": current_close  # At the candlestick level
            })

        markers.append(marker)

    # Convert data to JSON for JavaScript
    candlestick_json = json.dumps(candlestick_data)
    volume_json = json.dumps(volume_data)
    support_lower_json = json.dumps(support_lower_data)
    support_upper_json = json.dumps(support_upper_data)
    resistance_lower_json = json.dumps(resistance_lower_data)
    resistance_upper_json = json.dumps(resistance_upper_data)
    markers_json = json.dumps(markers)

    # HTML and JavaScript code to render TradingView Lightweight Charts with animation
    chart_html = f"""
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    </head>
    <body>
        <div id="chart" style="width: 100%; height: 600px;"></div>
        <script>
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                width: document.getElementById('chart').offsetWidth,
                height: 600,
                layout: {{
                    background: {{ color: '#090008' }},
                    textColor: '#FFFFFF',
                    fontSize: 16,
                    fontFamily: 'Helvetica'
                }},
                grid: {{
                    vertLines: {{ color: '#444' }},
                    horzLines: {{ color: '#444' }},
                }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                    barSpacing: 15,  // Increased spacing between candlesticks
                }},
                rightPriceScale: {{
                    scaleMargins: {{ top: 0.2, bottom: 0.2 }},  // Padding for markers
                }}
            }});

            // Add candlestick series
            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#00ff55',
                downColor: '#ed4807',
                borderUpColor: '#FFFFFF',
                borderDownColor: '#FFFFFF',
                wickUpColor: '#FFFFFF',
                wickDownColor: '#FFFFFF',
                barWidth: 0.8
            }});
            const candlestickData = {candlestick_json};

            // Add volume series
            const volumeSeries = chart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
                scaleMargins: {{ top: 0.8, bottom: 0 }}
            }});
            const volumeData = {volume_json};

            // Add support and resistance lines
            const supportLowerSeries = chart.addLineSeries({{ color: '#00FF55', lineWidth: 3, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'Support Lower' }});
            const supportLowerData = {support_lower_json};

            const supportUpperSeries = chart.addLineSeries({{ color: '#00FF55', lineWidth: 3, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'Support Upper' }});
            const supportUpperData = {support_upper_json};

            const resistanceLowerSeries = chart.addLineSeries({{ color: '#ED4807', lineWidth: 3, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'Resistance Lower' }});
            const resistanceLowerData = {resistance_lower_json};

            const resistanceUpperSeries = chart.addLineSeries({{ color: '#ED4807', lineWidth: 3, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'Resistance Upper' }});
            const resistanceUpperData = {resistance_upper_json};

            // Add markers
            const markers = {markers_json};

            // Add horizontal line at $200
            const horizontalLine = chart.addLineSeries({{ color: '#FFFFFF', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, price: 200 }});
            horizontalLine.createPriceLine({{
                price: 200,
                color: '#FFFFFF',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: true,
                title: '$200'
            }});

            // Add watermark
            chart.applyOptions({{
                watermark: {{
                    color: 'rgba(180, 180, 240, 0.7)',
                    visible: true,
                    text: '1D'
                }}
            }});

            // Enable legend
            chart.applyOptions({{
                layout: {{
                    fontSize: 14
                }}
            }});

            // Animation: Add data points one at a time
            let candlestickIndex = 0;
            let volumeIndex = 0;
            let supportLowerIndex = 0;
            let supportUpperIndex = 0;
            let resistanceLowerIndex = 0;
            let resistanceUpperIndex = 0;
            let markerIndex = 0;

            function updateChart() {{
                if (candlestickIndex < candlestickData.length) {{
                    candlestickSeries.update(candlestickData[candlestickIndex]);
                    candlestickIndex++;
                }}
                if (volumeIndex < volumeData.length) {{
                    volumeSeries.update(volumeData[volumeIndex]);
                    volumeIndex++;
                }}
                if (supportLowerIndex < supportLowerData.length) {{
                    supportLowerSeries.update(supportLowerData[supportLowerIndex]);
                    supportLowerIndex++;
                }}
                if (supportUpperIndex < supportUpperData.length) {{
                    supportUpperSeries.update(supportUpperData[supportUpperIndex]);
                    supportUpperIndex++;
                }}
                if (resistanceLowerIndex < resistanceLowerData.length) {{
                    resistanceLowerSeries.update(resistanceLowerData[resistanceLowerIndex]);
                    resistanceLowerIndex++;
                }}
                if (resistanceUpperIndex < resistanceUpperData.length) {{
                    resistanceUpperSeries.update(resistanceUpperData[resistanceUpperIndex]);
                    resistanceUpperIndex++;
                }}
                if (markerIndex < markers.length) {{
                    candlestickSeries.setMarkers(markers.slice(0, markerIndex + 1));
                    markerIndex++;
                }}
                if (candlestickIndex < candlestickData.length || 
                    volumeIndex < volumeData.length || 
                    supportLowerIndex < supportLowerData.length || 
                    supportUpperIndex < supportUpperData.length || 
                    resistanceLowerIndex < resistanceLowerData.length || 
                    resistanceUpperIndex < resistanceUpperData.length || 
                    markerIndex < markers.length) {{
                    setTimeout(updateChart, 100);
                }}
            }}
            updateChart();

            // Auto-resize chart on window resize
            window.addEventListener('resize', () => {{
                chart.resize(document.getElementById('chart').offsetWidth, 600);
            }});

            // Adjust time scale to fit all data
            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """

    # Render the chart using Streamlit's components.html
    components.html(chart_html, height=600)

# Chatbot function (unchanged)
def chatbot_interface():
    st.header("TSLA Data Chatbot")
    if not gemini_available:
        st.warning("Chartbot is disabled due to API key issues.")
        return

    st.write("Ask questions about the TSLA stock data (e.g., 'How many days in 2023 was TSLA bullish?')")

    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available for the chatbot.")
        return

    # Create a comprehensive data summary
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
    # Sample one row per year
    sample_rows = df.groupby('year').apply(lambda x: x.sample(1)).to_string()
    context = f"""You are a financial analyst assistant. You have access to TSLA stock data from 2022 to May 2025 with columns: {', '.join(df.columns)}. 
    Summary statistics:
    - Bullish days (close > open) per year:\n{bullish_days}
    - Signal counts per year (LONG, SHORT, None):\n{signal_counts}
    - Neutral signals (direction is None or empty) per year and month:\n{neutral_signals}
    - Price and volume stats per year:\n{price_stats}
    Sample data (one row per year):\n{sample_rows}
    Answer questions based on this data. Provide concise and accurate answers."""

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about TSLA data")
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        try:
            prompt = f"{context}\n\nQuestion: {user_input}"
            response = model.generate_content(prompt)
            answer = response.text

            # Append assistant response
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    # Example questions (updated for 2022–2025 data)
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

    # Load data for CSV download
    df = load_data()
    if not df.empty:
        # Add a download button for the CSV file
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="tsla_data.csv",
            mime="text/csv",
        )

    tab1, tab2 = st.tabs(["Candlestick Chart", "Chatbot"])

    with tab1:
        st.header("TSLA Candlestick Chart")
        render_chart()

    with tab2:
        chatbot_interface()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows multiprocessing
    main()
                                 #python -m streamlit run app.py