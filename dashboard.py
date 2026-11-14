"""
dashboard.py — Neo Brutalist Digital Twin Command Center for Indore Traffic RL
==============================================================================
Launches Thread 1 (RL/SUMO Engine) in a daemon thread and serves Thread 2 (Dash UI).
"""

import argparse
import threading
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from shared_state import (
    INTERSECTION_NAMES,
    update_config,
    get_config_snapshot,
    get_metrics_snapshot,
)
from train import start_training_loop

# ══════════════════════════════════════════════════════════════════════════════
# Neo Brutalism Visual Constants
# ══════════════════════════════════════════════════════════════════════════════
BG_PAPER = "#F4F4F0"
ACCENT_YELLOW = "#FFDE59"
ACCENT_PINK = "#FF3366"
ACCENT_CYAN = "#00E5FF"
ACCENT_LIME = "#76FF03"
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"

BRUTAL_CARD_STYLE = {
    "border": f"3px solid {PURE_BLACK}",
    "borderRadius": "0px",
    "boxShadow": f"6px 6px 0px {PURE_BLACK}",
    "padding": "14px",
    "fontFamily": "'Courier New', Courier, monospace",
    "color": PURE_BLACK,
}

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Dash App & Custom Neo Brutalism CSS
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="INDORE TRAFFIC RL // DIGITAL TWIN")

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
                border-radius: 0px !important;
                font-family: 'Courier New', Courier, monospace !important;
            }
            body {
                background-color: #F4F4F0;
                color: #000000;
                margin: 0;
                padding: 18px;
            }
            .brutal-btn {
                background-color: #FF3366;
                color: #FFFFFF;
                border: 3px solid #000000;
                box-shadow: 6px 6px 0px #000000;
                padding: 14px 16px;
                font-weight: 900;
                font-size: 15px;
                cursor: pointer;
                width: 100%;
                text-transform: uppercase;
                transition: transform 0.05s ease, box-shadow 0.05s ease;
            }
            .brutal-btn:hover {
                background-color: #FF0044;
            }
            .brutal-btn:active {
                transform: translate(4px, 4px);
                box-shadow: 2px 2px 0px #000000;
            }
            .brutal-input {
                width: 100%;
                padding: 8px 10px;
                border: 3px solid #000000;
                box-shadow: 4px 4px 0px #000000;
                background-color: #FFFFFF;
                font-weight: 900;
                font-size: 14px;
                color: #000000;
            }
            .rc-slider-rail {
                background-color: #FFFFFF !important;
                border: 2px solid #000000 !important;
                height: 10px !important;
            }
            .rc-slider-track {
                background-color: #000000 !important;
                height: 10px !important;
            }
            .rc-slider-handle {
                background-color: #FFDE59 !important;
                border: 3px solid #000000 !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -6px !important;
                opacity: 1 !important;
                box-shadow: 2px 2px 0px #000000 !important;
            }
            .rc-slider-mark-text {
                color: #000000 !important;
                font-weight: 800 !important;
                font-size: 11px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Dashboard Layout
# ══════════════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    dcc.Interval(id="telemetry-interval", interval=1000, n_intervals=0),

    # Top Title Header
    html.Div([
        html.Div([
            html.H1(
                "INDORE ARTERIAL GRID // RL DIGITAL TWIN COMMAND CENTER",
                style={"margin": "0", "fontSize": "22px", "fontWeight": "900", "letterSpacing": "-0.5px"},
            ),
            html.Div(
                "6-INTERSECTION SUMO + TRACI + STABLE-BASELINES3 PPO LIVE TELEMETRY & STEERING",
                style={"fontSize": "12px", "fontWeight": "700", "marginTop": "4px"},
            ),
        ]),
        html.Div(
            id="header-sim-status",
            style={
                "backgroundColor": PURE_WHITE,
                "border": f"3px solid {PURE_BLACK}",
                "boxShadow": f"4px 4px 0px {PURE_BLACK}",
                "padding": "8px 14px",
                "fontWeight": "900",
                "fontSize": "13px",
            },
        ),
    ], style={
        **BRUTAL_CARD_STYLE,
        "backgroundColor": ACCENT_YELLOW,
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "18px",
    }),

    # Main 2-Column Body: Left Control Sidebar + Right KPI & Chart Grid
    html.Div([
        # ── LEFT CONTROL SIDEBAR (LIVE STEERING INPUTS) ──
        html.Div([
            html.Div(
                "LIVE STEERING CONTROLS",
                style={
                    "fontSize": "16px",
                    "fontWeight": "900",
                    "borderBottom": f"3px solid {PURE_BLACK}",
                    "paddingBottom": "8px",
                    "marginBottom": "14px",
                },
            ),

            # 1. Brain Toggle Switch
            html.Div([
                html.Label("1. SIGNAL CONTROL BRAIN:", style={"fontWeight": "900", "fontSize": "12px"}),
                dcc.RadioItems(
                    id="brain-toggle",
                    options=[
                        {"label": " RL AGENT (PPO BRAIN)", "value": "RL"},
                        {"label": " FIXED-TIME TIMER", "value": "FIXED"},
                    ],
                    value="RL",
                    labelStyle={"display": "block", "marginTop": "8px", "fontWeight": "900", "cursor": "pointer"},
                    inputStyle={"marginRight": "8px", "transform": "scale(1.3)"},
                ),
                html.Div(
                    id="brain-status-badge",
                    style={
                        "marginTop": "10px",
                        "padding": "8px",
                        "border": f"3px solid {PURE_BLACK}",
                        "boxShadow": f"4px 4px 0px {PURE_BLACK}",
                        "fontWeight": "900",
                        "textAlign": "center",
                        "fontSize": "13px",
                    },
                ),
            ], style={"marginBottom": "20px"}),

            # 2. Traffic Density Slider
            html.Div([
                html.Label(
                    id="density-slider-label",
                    children="2. TRAFFIC DENSITY: 1.0x",
                    style={"fontWeight": "900", "fontSize": "12px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.Slider(
                    id="density-slider",
                    min=0.5,
                    max=3.0,
                    step=0.25,
                    value=1.0,
                    marks={0.5: "0.5x", 1.0: "1x", 2.0: "2x", 3.0: "3x"},
                ),
            ], style={"marginBottom": "24px"}),

            # 3. Target Episodes & Learning Rate Inputs
            html.Div([
                html.Label(
                    "3. TARGET EPISODES:",
                    style={"fontWeight": "900", "fontSize": "12px", "display": "block", "marginBottom": "4px"},
                ),
                dcc.Input(
                    id="target-episodes-input",
                    type="number",
                    value=50,
                    min=1,
                    max=10000,
                    step=1,
                    className="brutal-input",
                ),
                html.Label(
                    "4. PPO LEARNING RATE:",
                    style={"fontWeight": "900", "fontSize": "12px", "display": "block", "marginTop": "14px", "marginBottom": "4px"},
                ),
                dcc.Input(
                    id="learning-rate-input",
                    type="number",
                    value=0.0003,
                    min=0.00001,
                    max=0.05,
                    step=0.00005,
                    className="brutal-input",
                ),
                html.Div(
                    id="training-params-feedback",
                    style={"fontSize": "11px", "fontWeight": "800", "marginTop": "8px"},
                ),
            ], style={"marginBottom": "20px"}),

            # 5. Simulation Step Delay Slider
            html.Div([
                html.Label(
                    id="delay-slider-label",
                    children="5. STEP DELAY: 0 ms",
                    style={"fontWeight": "900", "fontSize": "12px", "display": "block", "marginBottom": "8px"},
                ),
                dcc.Slider(
                    id="step-delay-slider",
                    min=0,
                    max=500,
                    step=50,
                    value=0,
                    marks={0: "0ms", 250: "250ms", 500: "500ms"},
                ),
            ], style={"marginBottom": "26px"}),

            # 6. Dispatch Emergency Vehicle Push-Button
            html.Button(
                "DISPATCH EMERGENCY VEHICLE",
                id="dispatch-ev-btn",
                n_clicks=0,
                className="brutal-btn",
            ),
            html.Div(
                id="ev-button-feedback",
                style={"fontSize": "11px", "fontWeight": "900", "marginTop": "10px", "textAlign": "center"},
            ),
        ], style={
            **BRUTAL_CARD_STYLE,
            "backgroundColor": PURE_WHITE,
            "width": "300px",
            "flexShrink": "0",
        }),

        # ── RIGHT CONTENT AREA (TOP KPI BANNER + MAIN VISUALIZATION GRID) ──
        html.Div([
            # Top KPI Banner (4 Brutalist Metric Cards)
            html.Div([
                # Card 1: Emergency Response Status Card
                html.Div([
                    html.Div("EMERGENCY RESPONSE STATUS", style={"fontSize": "11px", "fontWeight": "900"}),
                    html.Div(id="kpi-ev-id", style={"fontSize": "18px", "fontWeight": "900", "marginTop": "4px"}),
                    html.Div(id="kpi-ev-junction", style={"fontSize": "12px", "fontWeight": "800", "marginTop": "4px"}),
                    html.Div(
                        id="kpi-ev-time-saved",
                        style={
                            "marginTop": "6px",
                            "padding": "4px 6px",
                            "backgroundColor": PURE_WHITE,
                            "border": f"2px solid {PURE_BLACK}",
                            "fontWeight": "900",
                            "fontSize": "12px",
                        },
                    ),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": ACCENT_PINK, "flex": "1"}),

                # Card 2: Live CO2 Emissions & Fuel Wasted
                html.Div([
                    html.Div("ENVIRONMENTAL TELEMETRY", style={"fontSize": "11px", "fontWeight": "900"}),
                    html.Div(id="kpi-co2-total", style={"fontSize": "20px", "fontWeight": "900", "marginTop": "4px"}),
                    html.Div(id="kpi-co2-rate", style={"fontSize": "12px", "fontWeight": "800", "marginTop": "4px"}),
                    html.Div(id="kpi-fuel-stats", style={"fontSize": "12px", "fontWeight": "900", "marginTop": "4px"}),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": ACCENT_YELLOW, "flex": "1"}),

                # Card 3: Network Throughput
                html.Div([
                    html.Div("NETWORK THROUGHPUT", style={"fontSize": "11px", "fontWeight": "900"}),
                    html.Div(id="kpi-throughput", style={"fontSize": "24px", "fontWeight": "900", "marginTop": "6px"}),
                    html.Div("CARS CLEARED / MIN", style={"fontSize": "11px", "fontWeight": "800", "marginTop": "4px"}),
                    html.Div(id="kpi-active-vehicles", style={"fontSize": "12px", "fontWeight": "900", "marginTop": "6px"}),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": ACCENT_CYAN, "flex": "1"}),

                # Card 4: Phase Flicker Rate
                html.Div([
                    html.Div("SIGNAL STABILITY", style={"fontSize": "11px", "fontWeight": "900"}),
                    html.Div(id="kpi-flicker-rate", style={"fontSize": "24px", "fontWeight": "900", "marginTop": "6px"}),
                    html.Div("PHASE SWITCHES / MIN", style={"fontSize": "11px", "fontWeight": "800", "marginTop": "4px"}),
                    html.Div("MIN GREEN LOCK: 10.0s", style={"fontSize": "12px", "fontWeight": "900", "marginTop": "6px"}),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": ACCENT_LIME, "flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "18px"}),

            # Main Visualization Grid
            # Row 1: Chart 1 (Full Width Live Global Queue Length with RL vs Fixed Shading)
            html.Div([
                html.Div(
                    "CHART 1 // LIVE GLOBAL QUEUE LENGTH (RL AGENT VS. FIXED-TIME SHADING)",
                    style={"fontWeight": "900", "fontSize": "13px", "marginBottom": "6px"},
                ),
                dcc.Graph(id="chart-global-queue", config={"displayModeBar": False}, style={"height": "260px"}),
            ], style={**BRUTAL_CARD_STYLE, "backgroundColor": PURE_WHITE, "marginBottom": "18px"}),

            # Row 2: Chart 2 (6-Intersection Queue Grid) & Chart 3 (Live Cumulative RL Reward Curve)
            html.Div([
                html.Div([
                    html.Div(
                        "CHART 2 // 6-INTERSECTION REAL-TIME QUEUE CONGESTION GRID",
                        style={"fontWeight": "900", "fontSize": "13px", "marginBottom": "6px"},
                    ),
                    dcc.Graph(id="chart-intersection-grid", config={"displayModeBar": False}, style={"height": "260px"}),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": PURE_WHITE, "flex": "1"}),

                html.Div([
                    html.Div(
                        "CHART 3 // LIVE CUMULATIVE RL REWARD CONVERGENCE CURVE",
                        style={"fontWeight": "900", "fontSize": "13px", "marginBottom": "6px"},
                    ),
                    dcc.Graph(id="chart-reward-curve", config={"displayModeBar": False}, style={"height": "260px"}),
                ], style={**BRUTAL_CARD_STYLE, "backgroundColor": PURE_WHITE, "flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}),
        ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
    ], style={"display": "flex", "gap": "18px", "alignItems": "flex-start"}),
])


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks: UI Controls -> Shared global_config (Immediate Two-Way Steering)
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("brain-status-badge", "children"),
    Output("brain-status-badge", "style"),
    Input("brain-toggle", "value"),
)
def cb_set_control_mode(mode):
    update_config("control_mode", mode)
    base_style = {
        "marginTop": "10px",
        "padding": "8px",
        "border": f"3px solid {PURE_BLACK}",
        "boxShadow": f"4px 4px 0px {PURE_BLACK}",
        "fontWeight": "900",
        "textAlign": "center",
        "fontSize": "13px",
    }
    if mode == "RL":
        base_style["backgroundColor"] = ACCENT_LIME
        base_style["color"] = PURE_BLACK
        return "ACTIVE BRAIN: PPO RL AGENT", base_style
    else:
        base_style["backgroundColor"] = ACCENT_PINK
        base_style["color"] = PURE_WHITE
        return "ACTIVE BRAIN: FIXED TIMER", base_style


@app.callback(
    Output("density-slider-label", "children"),
    Input("density-slider", "value"),
)
def cb_set_density(val):
    update_config("traffic_density", float(val))
    return f"2. TRAFFIC DENSITY: {val:.2f}x"


@app.callback(
    Output("training-params-feedback", "children"),
    Input("target-episodes-input", "value"),
    Input("learning-rate-input", "value"),
)
def cb_set_training_params(target_eps, lr):
    if target_eps is not None:
        update_config("target_episodes", int(target_eps))
    if lr is not None:
        update_config("learning_rate", float(lr))
    return f"LIVE SYNCED: Max Ep={target_eps} | LR={lr}"


@app.callback(
    Output("delay-slider-label", "children"),
    Input("step-delay-slider", "value"),
)
def cb_set_step_delay(val):
    update_config("step_delay_ms", int(val))
    return f"5. STEP DELAY: {val} ms"


@app.callback(
    Output("ev-button-feedback", "children"),
    Input("dispatch-ev-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cb_dispatch_ev(n_clicks):
    update_config("spawn_ev", True)
    return f"SIREN TRIGGERED (#{n_clicks}) -> SPAWNING NEXT STEP"


# ══════════════════════════════════════════════════════════════════════════════
# Callback: Interval Pull -> Update KPI Cards & 3 Live Plotly Charts (Every 1s)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_brutal_chart_layout(fig, yaxis_title, xaxis_title="SIMULATION TIME (s)"):
    fig.update_layout(
        template="none",
        paper_bgcolor=PURE_WHITE,
        plot_bgcolor=BG_PAPER,
        font=dict(family="Courier New, monospace", color=PURE_BLACK, size=11),
        margin=dict(l=48, r=16, t=12, b=38),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(weight="bold")),
            showgrid=True,
            gridcolor="#D0D0CC",
            linecolor=PURE_BLACK,
            linewidth=2,
            mirror=True,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(weight="bold")),
            showgrid=True,
            gridcolor="#D0D0CC",
            linecolor=PURE_BLACK,
            linewidth=2,
            mirror=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=ACCENT_YELLOW,
            bordercolor=PURE_BLACK,
            borderwidth=2,
        ),
    )
    return fig


@app.callback(
    Output("header-sim-status", "children"),
    Output("kpi-ev-id", "children"),
    Output("kpi-ev-junction", "children"),
    Output("kpi-ev-time-saved", "children"),
    Output("kpi-co2-total", "children"),
    Output("kpi-co2-rate", "children"),
    Output("kpi-fuel-stats", "children"),
    Output("kpi-throughput", "children"),
    Output("kpi-active-vehicles", "children"),
    Output("kpi-flicker-rate", "children"),
    Output("chart-global-queue", "figure"),
    Output("chart-intersection-grid", "figure"),
    Output("chart-reward-curve", "figure"),
    Input("telemetry-interval", "n_intervals"),
)
def cb_update_telemetry(_):
    m = get_metrics_snapshot()
    cfg = get_config_snapshot()

    # Header Status
    header_text = (
        f"EPISODE: {m['episode']}/{cfg['target_episodes']} | "
        f"STEP: {m['episode_step']}/720 | "
        f"SIM TIME: {int(m['sim_time'])}s | "
        f"MODE: {m['current_mode']}"
    )

    # KPI 1: Emergency Vehicle
    if m["ev_active"]:
        ev_id_txt = f"ACTIVE: {m['ev_id']}"
        ev_junc_txt = f"NEXT: {m['ev_junction']}"
    else:
        ev_id_txt = f"STANDBY (TOTAL: {m['ev_total_dispatched']})"
        ev_junc_txt = f"LAST: {m['ev_last_completed_id']}"
    ev_saved_txt = f"TIME SAVED VS NORMAL: {m['ev_time_saved_s']:.1f}s"

    # KPI 2: Environmental
    co2_total_txt = f"CO2: {m['co2_total_kg']:.2f} kg"
    co2_rate_txt = f"RATE: {m['co2_rate_gps']:.1f} g/s"
    fuel_txt = f"FUEL: {m['fuel_total_l']:.2f} L ({m['fuel_rate_mlps']:.1f} mL/s)"

    # KPI 3 & 4: Throughput & Flicker
    throughput_txt = f"{m['throughput_per_min']:.1f}"
    active_veh_txt = f"ACTIVE IN NETWORK: {m['active_vehicles']} VEH"
    flicker_txt = f"{m['phase_flicker_rate']:.1f}"

    # ── CHART 1: Live Global Queue Length (with RL vs FIXED Mode Color-Coding) ──
    fig_queue = go.Figure()
    ts = m["history_timestamps"]
    qs = m["history_global_queue"]
    modes = m["history_mode"]

    if ts:
        # Base connecting trace
        fig_queue.add_trace(go.Scatter(
            x=ts, y=qs, mode="lines",
            line=dict(color=PURE_BLACK, width=3),
            name="Global Halting Queue",
            showlegend=False,
        ))
        # Overlay color-coded markers for RL vs FIXED mode
        rl_x = [t for t, md in zip(ts, modes) if md == "RL"]
        rl_y = [q for q, md in zip(qs, modes) if md == "RL"]
        fx_x = [t for t, md in zip(ts, modes) if md == "FIXED"]
        fx_y = [q for q, md in zip(qs, modes) if md == "FIXED"]

        if rl_x:
            fig_queue.add_trace(go.Scatter(
                x=rl_x, y=rl_y, mode="markers",
                marker=dict(color=ACCENT_LIME, size=8, line=dict(color=PURE_BLACK, width=2)),
                name="RL Agent Mode",
            ))
        if fx_x:
            fig_queue.add_trace(go.Scatter(
                x=fx_x, y=fx_y, mode="markers",
                marker=dict(color=ACCENT_PINK, size=8, symbol="square", line=dict(color=PURE_BLACK, width=2)),
                name="Fixed-Time Mode",
            ))
    _apply_brutal_chart_layout(fig_queue, yaxis_title="STOPPED VEHICLES")

    # ── CHART 2: 6-Intersection Congestion Bar Grid (Green to Gridlock Red) ──
    int_names_short = ["J1: MR10", "J2: Ring S", "J3: Ring W", "J4: VijayNgr", "J5: Radisson", "J6: BombayH"]
    int_counts = [m["intersection_queues"].get(full_name, 0) for full_name in INTERSECTION_NAMES]
    bar_colors = []
    for c in int_counts:
        if c < 8:
            bar_colors.append(ACCENT_LIME)
        elif c < 20:
            bar_colors.append(ACCENT_YELLOW)
        else:
            bar_colors.append(ACCENT_PINK)

    fig_grid = go.Figure(go.Bar(
        x=int_names_short,
        y=int_counts,
        text=int_counts,
        textposition="auto",
        marker=dict(color=bar_colors, line=dict(color=PURE_BLACK, width=3)),
    ))
    _apply_brutal_chart_layout(fig_grid, yaxis_title="QUEUE COUNT", xaxis_title="6 SIGNALIZED INTERSECTIONS")

    # ── CHART 3: Live Cumulative RL Reward Curve ──
    fig_reward = go.Figure()
    cum_rew = m["history_cum_reward"]
    if ts and cum_rew:
        fig_reward.add_trace(go.Scatter(
            x=ts,
            y=cum_rew,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 229, 255, 0.25)",
            line=dict(color=PURE_BLACK, width=3),
            name="Cumulative Episode Reward",
        ))
    _apply_brutal_chart_layout(fig_reward, yaxis_title="CUMULATIVE REWARD")

    return (
        header_text,
        ev_id_txt,
        ev_junc_txt,
        ev_saved_txt,
        co2_total_txt,
        co2_rate_txt,
        fuel_txt,
        throughput_txt,
        active_veh_txt,
        flicker_txt,
        fig_queue,
        fig_grid,
        fig_reward,
    )


# ══════════════════════════════════════════════════════════════════════════════
# System Entry Point: Starts Thread 1 (RL/SUMO) & Thread 2 (Dash Web Server)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Indore RL Digital Twin Command Center")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO-GUI alongside the Dash Command Center")
    parser.add_argument("--port", type=int, default=8050, help="Port for Plotly Dash web server")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total RL timesteps to run")
    args = parser.parse_args()

    # Thread 1: Background Daemon Thread running RL Agent & SUMO Engine
    sim_thread = threading.Thread(
        target=start_training_loop,
        kwargs={"use_gui": args.gui, "total_timesteps": args.timesteps},
        daemon=True,
    )
    sim_thread.start()

    print("\n" + "=" * 72)
    print(f"  NEO BRUTALIST DIGITAL TWIN COMMAND CENTER RUNNING AT: http://127.0.0.1:{args.port}")
    print("=" * 72 + "\n")

    # Thread 2: Main Thread running Plotly Dash Server (use_reloader=False prevents duplicate SUMO threads)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)