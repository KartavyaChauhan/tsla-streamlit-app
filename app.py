import pandas as pd
import streamlit as st
import ast
import plotly.graph_objects as go
import google.generativeai as genai
import os

# Data Preparation
df = pd.read_csv("data/tsla.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# UI Setup
tab1, tab2, tab3 = st.tabs(["📊 Chart", "🤖 Gemini Chatbot", "🎬 Animation"])

# Chart Tab
with tab1:
    st.title("TSLA Dashboard")

    date_range_option = st.selectbox(
        "Select Date Range for Chart",
        ["August 2022 (Sample)", "Full Range (2022 to 2025)"],
        index=0
    )

    if date_range_option == "August 2022 (Sample)":
        start_date = "2022-08-25"
        end_date = "2022-08-31"
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        yaxis_range = [0, 350]
    else:
        filtered_df = df
        yaxis_range = [0, 400]

    st.dataframe(df[['timestamp', 'direction', 'Support', 'Resistance', 'open', 'high', 'low', 'close', 'volume']].head())

    # Safely parse Support and Resistance columns
    def safe_literal_eval(value):
        # Handle NaN, None, or non-string values
        if not isinstance(value, str) or value in [None, 'nan', '']:
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []  # Return empty list for invalid entries

    filtered_df["Support"] = filtered_df["Support"].apply(safe_literal_eval)
    filtered_df["Resistance"] = filtered_df["Resistance"].apply(safe_literal_eval)

    support_lower = []
    support_upper = []
    resistance_lower = []
    resistance_upper = []

    for _, row in filtered_df.iterrows():
        time = row["timestamp"].strftime('%Y-%m-%d')
        if row["Support"] and isinstance(row["Support"], list) and len(row["Support"]) > 0:
            support_lower.append((time, min(row["Support"])))
            support_upper.append((time, max(row["Support"])))
        if row["Resistance"] and isinstance(row["Resistance"], list) and len(row["Resistance"]) > 0:
            resistance_lower.append((time, min(row["Resistance"])))
            resistance_upper.append((time, max(row["Resistance"])))

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=filtered_df['timestamp'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='OHLC'
    ))

    fig.add_trace(go.Scatter(
        x=[x[0] for x in support_lower],
        y=[x[1] for x in support_lower],
        mode='lines',
        name='Support',
        line=dict(color='green', width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=[x[0] for x in support_upper],
        y=[x[1] for x in support_upper],
        mode='lines',
        name='Support Band',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(color='green', width=1),
        opacity=0.3
    ))

    fig.add_trace(go.Scatter(
        x=[x[0] for x in resistance_lower],
        y=[x[1] for x in resistance_lower],
        mode='lines',
        name='Resistance',
        line=dict(color='red', width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=[x[0] for x in resistance_upper],
        y=[x[1] for x in resistance_upper],
        mode='lines',
        name='Resistance Band',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='red', width=1),
        opacity=0.3
    ))

    for _, row in filtered_df.iterrows():
        direction = row["direction"]
        if pd.isna(direction):
            fig.add_annotation(
                x=row["timestamp"],
                y=(row["high"] + row["low"]) / 2,
                text="None",
                showarrow=True,
                arrowhead=0,
                arrowcolor="yellow",
                font=dict(color="yellow")
            )
        elif direction == "SHORT":
            fig.add_annotation(
                x=row["timestamp"],
                y=row["high"] + 5,
                text="SHORT",
                showarrow=True,
                arrowhead=1,
                arrowcolor="red",
                font=dict(color="red")
            )
        elif direction == "LONG":
            fig.add_annotation(
                x=row["timestamp"],
                y=row["low"] - 5,
                text="LONG",
                showarrow=True,
                arrowhead=1,
                arrowcolor="green",
                font=dict(color="green")
            )

    fig.update_layout(
        title="TSLA Candlestick Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        plot_bgcolor="#222",
        paper_bgcolor="#222",
        font=dict(color="#DDD"),
        showlegend=True,
        yaxis=dict(range=yaxis_range),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.write("Use the slider below the chart to zoom in on specific date ranges.")
    st.plotly_chart(fig, use_container_width=True)

# Chatbot Tab
with tab2:
    st.title("Gemini Chatbot for TSLA Data")

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"Error setting up Gemini API: {str(e)}")
        st.write("Please ensure your GEMINI_API_KEY is set correctly in your environment.")
        st.stop()

    prompt = st.text_input("Ask about TSLA data (e.g., 'How many days in 2023 was TSLA bullish?'):")

    st.write("**Suggested Questions to Try:**")
    st.write("- What was the lowest closing price of TSLA in 2024?")
    st.write("- How many days in 2022 did TSLA have a volume above 150?")
    st.write("- What was the biggest daily price change in 2023?")
    st.write("- On which days in 2024 did TSLA have a closing price above 300?")
    st.write("- Summarize TSLA's performance in 2022.")

    if prompt:
        relevant_df = df
        data_summary = ""

        years = [str(year) for year in range(2022, 2026)]
        mentioned_year = None
        for year in years:
            if year in prompt:
                mentioned_year = year
                break

        if mentioned_year:
            relevant_df = df[df['timestamp'].dt.year == int(mentioned_year)]
            st.write(f"Filtering data for {mentioned_year}...")

        columns_to_include = ['timestamp', 'direction', 'open', 'close', 'high', 'low', 'volume']
        
        if "bullish" in prompt.lower() or "closing price" in prompt.lower():
            columns_to_include = ['timestamp', 'open', 'close']
        elif "volume" in prompt.lower():
            columns_to_include = ['timestamp', 'volume']
        elif "short" in prompt.lower() or "long" in prompt.lower():
            columns_to_include = ['timestamp', 'direction']
        elif "price change" in prompt.lower():
            columns_to_include = ['timestamp', 'high', 'low']
        elif "summarize" in prompt.lower():
            columns_to_include = ['timestamp', 'open', 'close', 'volume']

        if not relevant_df.empty:
            data_summary = f"TSLA stock data for {mentioned_year if mentioned_year else 'all years'}:\n"
            for _, row in relevant_df.iterrows():
                row_summary = []
                for col in columns_to_include:
                    if col == 'timestamp':
                        row_summary.append(f"Date: {row['timestamp'].strftime('%Y-%m-%d')}")
                    elif col == 'direction':
                        row_summary.append(f"Direction: {row['direction'] if not pd.isna(row['direction']) else 'None'}")
                    else:
                        row_summary.append(f"{col.capitalize()}: {row[col]}")
                data_summary += ", ".join(row_summary) + "\n"
        else:
            data_summary = "No data available for the requested period."

        full_prompt = f"The following is TSLA stock data:\n{data_summary}\n\nUser question: {prompt}\nAnswer the question based on the data."

        try:
            response = model.generate_content(full_prompt)
            st.write("**Gemini Response:**")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.write("Please try rephrasing your question or check your API key.")

# Animation Tab
with tab3:
    st.title("TSLA Animation Replay")

    date_range_option = st.selectbox(
        "Select Date Range for Animation",
        ["August 2022 (Sample)", "2022", "2023", "2024", "2025 (Jan-May)", "Full Range (2022 to 2025)"],
        index=0
    )

    if date_range_option == "August 2022 (Sample)":
        start_date = "2022-08-25"
        end_date = "2022-08-31"
        anim_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        yaxis_range = [0, 350]
    elif date_range_option == "2022":
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        anim_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        yaxis_range = [0, 400]
    elif date_range_option == "2023":
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        anim_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        yaxis_range = [0, 400]
    elif date_range_option == "2024":
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        anim_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        yaxis_range = [0, 400]
    elif date_range_option == "2025 (Jan-May)":
        start_date = "2025-01-01"
        end_date = "2025-05-31"
        anim_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
        yaxis_range = [0, 400]
    else:
        anim_df = df.copy()
        yaxis_range = [0, 400]

    if len(anim_df) > 100:
        st.warning("Large date ranges may slow down the animation due to the number of frames. Consider selecting a smaller range for better performance.")

    # Safely parse Support and Resistance columns
    anim_df["Support"] = anim_df["Support"].apply(safe_literal_eval)
    anim_df["Resistance"] = anim_df["Resistance"].apply(safe_literal_eval)

    frames = []
    for i in range(len(anim_df)):
        frame_df = anim_df.iloc[:i+1]
        support_lower = []
        support_upper = []
        resistance_lower = []
        resistance_upper = []

        for _, row in frame_df.iterrows():
            time = row["timestamp"].strftime('%Y-%m-%d')
            if row["Support"] and isinstance(row["Support"], list) and len(row["Support"]) > 0:
                support_lower.append((time, min(row["Support"])))
                support_upper.append((time, max(row["Support"])))
            if row["Resistance"] and isinstance(row["Resistance"], list) and len(row["Resistance"]) > 0:
                resistance_lower.append((time, min(row["Resistance"])))
                resistance_upper.append((time, max(row["Resistance"])))

        frame = go.Frame(
            data=[
                go.Candlestick(
                    x=frame_df['timestamp'],
                    open=frame_df['open'],
                    high=frame_df['high'],
                    low=frame_df['low'],
                    close=frame_df['close'],
                    name='OHLC'
                ),
                go.Scatter(
                    x=[x[0] for x in support_lower],
                    y=[x[1] for x in support_lower],
                    mode='lines',
                    name='Support',
                    line=dict(color='green', width=1),
                    opacity=0.3
                ),
                go.Scatter(
                    x=[x[0] for x in support_upper],
                    y=[x[1] for x in support_upper],
                    mode='lines',
                    name='Support Band',
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    line=dict(color='green', width=1),
                    opacity=0.3
                ),
                go.Scatter(
                    x=[x[0] for x in resistance_lower],
                    y=[x[1] for x in resistance_lower],
                    mode='lines',
                    name='Resistance',
                    line=dict(color='red', width=1),
                    opacity=0.3
                ),
                go.Scatter(
                    x=[x[0] for x in resistance_upper],
                    y=[x[1] for x in resistance_upper],
                    mode='lines',
                    name='Resistance Band',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='red', width=1),
                    opacity=0.3
                )
            ],
            layout=dict(
                annotations=[
                    dict(
                        x=row["timestamp"],
                        y=(row["high"] + row["low"]) / 2 if pd.isna(row["direction"])
                        else row["high"] + 5 if row["direction"] == "SHORT"
                        else row["low"] - 5,
                        text="None" if pd.isna(row["direction"])
                        else row["direction"],
                        showarrow=True,
                        arrowhead=0 if pd.isna(row["direction"]) else 1,
                        arrowcolor="yellow" if pd.isna(row["direction"])
                        else "red" if row["direction"] == "SHORT"
                        else "green",
                        font=dict(
                            color="yellow" if pd.isna(row["direction"])
                            else "red" if row["direction"] == "SHORT"
                            else "green"
                        )
                    ) for _, row in frame_df.iterrows()
                ]
            )
        )
        frames.append(frame)

    fig_anim = go.Figure(
        data=[
            go.Candlestick(
                x=anim_df['timestamp'][:1],
                open=anim_df['open'][:1],
                high=anim_df['high'][:1],
                low=anim_df['low'][:1],
                close=anim_df['close'][:1],
                name='OHLC'
            ),
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Support',
                line=dict(color='green', width=1),
                opacity=0.3
            ),
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Support Band',
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(color='green', width=1),
                opacity=0.3
            ),
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Resistance',
                line=dict(color='red', width=1),
                opacity=0.3
            ),
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Resistance Band',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='red', width=1),
                opacity=0.3
            )
        ],
        layout=go.Layout(
            title="TSLA Animation Replay",
            yaxis_title="Price",
            xaxis_title="Date",
            plot_bgcolor="#222",
            paper_bgcolor="#222",
            font=dict(color="#DDD"),
            showlegend=True,
            yaxis=dict(range=yaxis_range),
            xaxis=dict(type="date"),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode="immediate")]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        ),
        frames=frames
    )

    st.plotly_chart(fig_anim, use_container_width=True)    
                     #python -m streamlit run app.py