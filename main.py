import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Function to load the ECG data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, delim_whitespace=True)
    return df

# Function for moving average
def moving_average(signal, window):
    n_data = len(signal)
    mav_1 = np.zeros(n_data * 2)
    mav_2 = np.zeros(n_data * 2)

    for i in range(window):
        signal[-i] = signal[0]

    for i in range(n_data):
        count = 0
        for j in range(window):
            count += signal[i - j]
        mav_1[i] = count / window

    for i in range(n_data, n_data + window):
        mav_1[i] = mav_1[n_data]

    for i in range(n_data, 0, -1):
        count = 0
        for j in range(window):
            count += mav_1[i + j]
        mav_2[i] = count / window

    mav_2 = np.array(mav_2[:n_data])
    return mav_2

# Function for DFT
def dft(signal):
    n_data = len(signal)
    X_real = np.zeros(n_data * 2)
    X_imaj = np.zeros(n_data * 2)
    MagDFT = np.zeros(n_data * 2)

    for k in range(n_data):
        for n in range(n_data):
            X_real[k] += (signal[n]) * np.cos(2 * np.pi * k * n / n_data)
            X_imaj[k] += (signal[n]) * np.sin(2 * np.pi * k * n / n_data)

    for k in range(n_data):
        MagDFT[k] = np.sqrt(np.square(X_real[k]) + np.square(X_imaj[k]))

    return MagDFT

# Streamlit interface
st.title("ECG Signal Processing")

ecg_file = "data_ecg.csv"

df = load_data(ecg_file)

time_index = df[df.columns[0]]
data = df[df.columns[1]]
data = np.array(data - 1.25)

# Display the raw ECG signal
st.subheader("Raw ECG Signal")
fig = go.Figure(data=go.Scatter(x=time_index, y=data, mode="lines"))
fig.update_layout(
    title="Raw Signal",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

# Moving Average
mav_result = moving_average(data, 18)
st.subheader("Moving Average Filter")
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=data, mode="lines", name="Original Signal"))
fig.add_trace(go.Scatter(x=time_index, y=mav_result, mode="lines", name="After MAV"))
fig.update_layout(
    title="Moving Average",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

# DFT of the signal
raw_data_dft = dft(data)
n_data = len(data)
fs = 1000
k = np.arange(0, n_data, 1, dtype=int)
fig = go.Figure(data=go.Scatter(x=k * fs / n_data, y=raw_data_dft[k], mode="lines"))
fig.update_layout(
    title="DFT of Raw Signal",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

# Apply Bandpass Filtering (LPF and HPF)
def butterworth_lpf(signal, freq_cutoff, orde):
    omega_c = 2 * np.pi * freq_cutoff
    sampling_period = 1 / fs
    sampling_period_squared = np.square(sampling_period)
    omega_c_squared = np.square(omega_c)
    y = np.zeros(len(signal))

    if orde == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = omega_c * signal[n] / ((2 / sampling_period) + omega_c)
            else:
                y[n] = (
                    ((2 / sampling_period) - omega_c) * y[n - 1]
                    + omega_c * signal[n]
                    + omega_c * signal[n - 1]
                ) / ((2 / sampling_period) + omega_c)
    elif orde == 2:
        for n in range(len(signal)):
            if n < 2:
                y[n] = (omega_c_squared * signal[n]) / (
                    (4 / sampling_period_squared)
                    + (2 * np.sqrt(2) * omega_c / sampling_period)
                    + omega_c_squared
                )
            else:
                y[n] = (
                    ((8 / sampling_period_squared) - 2 * omega_c_squared) * y[n - 1]
                    - (
                        (4 / sampling_period_squared)
                        - (2 * np.sqrt(2) * omega_c / sampling_period)
                        + omega_c_squared
                    )
                    * y[n - 2]
                    + omega_c_squared * signal[n]
                    + 2 * omega_c_squared * signal[n - 1]
                    + omega_c_squared * signal[n - 2]
                ) / (
                    (4 / sampling_period_squared)
                    + (2 * np.sqrt(2) * omega_c / sampling_period)
                    + omega_c_squared
                )
    return y

lpf_result = butterworth_lpf(mav_result, 20, 2)
st.subheader("Low Pass Filtered Signal")
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=mav_result, mode="lines", name="Before LPF"))
fig.add_trace(go.Scatter(x=time_index, y=lpf_result, mode="lines", name="After LPF"))
fig.update_layout(
    title="Low Pass Filtering",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

# High Pass Filtering
def butterworth_hpf(signal, freq_cutoff, orde):
    omega_c = 2 * np.pi * freq_cutoff
    sampling_period = 1 / fs
    sampling_period_squared = np.square(sampling_period)
    omega_c_squared = np.square(omega_c)
    y = np.zeros(len(signal))

    if orde == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = ((2 / sampling_period) * signal[n]) / (
                    (2 / sampling_period) + omega_c
                )
            else:
                y[n] = (
                    ((2 / sampling_period) * signal[n])
                    - ((2 / sampling_period) * signal[n - 1])
                    - ((omega_c - (2 / sampling_period)) * y[n - 1])
                ) / ((2 / sampling_period) + omega_c)
    elif orde == 2:
        for n in range(len(signal)):
            if n < 2:
                y[n] = (
                    (4 / sampling_period_squared)
                    * signal[n]
                    / (
                        omega_c_squared
                        + (2 * np.sqrt(2) * omega_c / sampling_period)
                        + (4 / sampling_period_squared)
                    )
                )
            else:
                y[n] = (
                    (4 / sampling_period_squared) * signal[n]
                    - (8 / sampling_period_squared) * signal[n - 1]
                    + (4 / sampling_period_squared) * signal[n - 2]
                    - (2 * omega_c - (8 / sampling_period_squared)) * y[n - 1]
                    - (
                        omega_c_squared
                        - (2 * np.sqrt(2) * omega_c / sampling_period)
                        + (4 / sampling_period_squared)
                    )
                    * y[n - 2]
                ) / (
                    omega_c_squared
                    + 2 * np.sqrt(2) * omega_c / sampling_period
                    + (4 / sampling_period_squared)
                )
    return y

hpf_result = butterworth_hpf(lpf_result, 4, 2)
st.subheader("High Pass Filtered Signal")
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=lpf_result, mode="lines", name="Before HPF"))
fig.add_trace(go.Scatter(x=time_index, y=hpf_result, mode="lines", name="After HPF"))
fig.update_layout(
    title="High Pass Filtering",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

filtered_signal_dft = dft(hpf_result)
n = np.arange(0, n_data, 1, dtype=int)
k = np.arange(0, n_data, 1, dtype=int)
fig = go.Figure(
    data=go.Scatter(x=k * fs / n_data, y=filtered_signal_dft[k], mode="lines")
)
fig.update_layout(
    title="DFT Filtered Signal",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

# SQUARING
squared_signal = []
for i in range(len(hpf_result)):
    squared_signal.append(np.square(hpf_result[i]))
mav_squared = moving_average(squared_signal, 25)
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=squared_signal, mode="lines", name="Squared"))
fig.add_trace(go.Scatter(x=time_index, y=mav_squared, mode="lines", name="MAV"))
fig.update_layout(
    title="Squaring and MAV",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=1000,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

threshold_result = []
for i in range(len(mav_squared)):
    if mav_squared[i] > (0.1 * max(mav_squared)):
        threshold_result.append(np.max(mav_result))
    else:
        threshold_result.append(0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_index, y=mav_result, mode="lines", name="Signal"))
fig.add_trace(
    go.Scatter(x=time_index, y=threshold_result, mode="lines", name="Threshold")
)
fig.update_layout(
    title="Thresholding Process",
    xaxis_title="Time (ms)",
    yaxis_title="Amplitude (mV)",
    width=800,
    height=500,
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True),
)
st.plotly_chart(fig)

peaks = 0
for i in range(len(threshold_result)):
    if threshold_result[i] > threshold_result[i - 1]:
        peaks += 1

time = n_data / fs
heart_rate = int(60 * peaks / time)
st.subheader(f"Heart Rate = {heart_rate} bpm")
