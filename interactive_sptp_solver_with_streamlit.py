import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="SPTP Solver", page_icon="🚚")

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #f0f2f6; /* Lighter background */
    }
    /* Card Styling */
    .card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
    }
    /* Metric Styling */
    .stMetric {
        border-radius: 0.5rem;
        padding: 0.75rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# --- CORE HEURISTIC LOGIC ---

def get_route_timeline(route, nodes, travel_times):
    """Calculates the detailed timeline, total trip time, and total wait time for a given route."""
    if not route or len(route) < 2:
        return {"details": [], "total_time": 0, "total_wait_time": 0}

    details = []
    vehicle_clock = 0.0
    production_clock = 0.0
    last_location = 0
    total_wait_time = 0.0

    details.append(f"T= {vehicle_clock:<5.1f} | 🏁 Start Trip at Depot")

    for i in range(1, len(route) - 1):
        current_location = route[i]
        travel = travel_times[last_location][current_location]

        arrival_time = vehicle_clock + travel
        production_finish_time = production_clock + nodes[current_location]['p']

        wait_time = max(0, production_finish_time - arrival_time)
        delivery_completion_time = arrival_time + wait_time

        details.append(f"         | ➡️ Travel from C{last_location} to C{current_location} ({travel} min)")
        details.append(f"T= {arrival_time:<5.1f} | 📍 Arrive at C{current_location}")
        details.append(f"         | 🏭 Production for C{current_location} finishes at T={production_finish_time:.1f}")

        if wait_time > 0:
            details.append(f"         | ⏳ WAIT for {wait_time:.1f} min")

        details.append(f"T= {delivery_completion_time:<5.1f} | ✅ Delivery to C{current_location} complete")

        total_wait_time += wait_time
        vehicle_clock = delivery_completion_time
        production_clock = production_finish_time
        last_location = current_location

    return_travel = travel_times[last_location][0]
    final_time = vehicle_clock + return_travel

    details.append(f"         | ➡️ Travel from C{last_location} to Depot ({return_travel} min)")
    details.append(f"T= {final_time:<5.1f} | 🏁 End Trip at Depot")

    return {
        "details": details,
        "total_time": final_time,
        "total_wait_time": total_wait_time
    }


def construct_initial_route(nodes, travel_times):
    """Builds an initial 'smart' route using a nearest neighbor heuristic."""
    customer_ids = [int(k) for k in nodes.keys() if k != 0]
    route = [0]
    unvisited = set(customer_ids)

    last_location = 0
    production_clock = 0.0
    vehicle_clock = 0.0

    while unvisited:
        best_next_node = -1
        min_wait_time = float('inf')
        best_travel_time = float('inf')

        for node_id in unvisited:
            travel = travel_times[last_location][node_id]
            arrival_time = vehicle_clock + travel
            production_finish_time = production_clock + nodes[node_id]['p']
            wait_time = max(0, production_finish_time - arrival_time)

            if wait_time < min_wait_time:
                min_wait_time = wait_time
                best_next_node = node_id
                best_travel_time = travel
            elif wait_time == min_wait_time and travel < best_travel_time:
                best_next_node = node_id
                best_travel_time = travel

        travel = travel_times[last_location][best_next_node]
        arrival = vehicle_clock + travel
        prod_finish = production_clock + nodes[best_next_node]['p']

        vehicle_clock = max(arrival, prod_finish)
        production_clock = prod_finish

        route.append(best_next_node)
        last_location = best_next_node
        unvisited.remove(best_next_node)

    route.append(0)
    return route


def improve_route(route, nodes, travel_times):
    """Improves a given route using the two-opt heuristic."""
    best_route = route[:]
    best_time = get_route_timeline(best_route, nodes, travel_times)['total_time']

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                new_time = get_route_timeline(new_route, nodes, travel_times)['total_time']

                if new_time < best_time:
                    best_route = new_route
                    best_time = new_time
                    improved = True

    return best_route


# --- VISUALIZATION ---

def plot_routes(initial_route, improved_route, nodes):
    """Plots both initial and improved routes on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    # Define high-contrast colors
    depot_color = '#1e293b'
    customer_color = '#4f46e5'
    initial_route_color = '#0ea5e9'
    improved_route_color = '#16a34a'

    # Plot nodes and annotations
    ax.plot(nodes[0]['x'], nodes[0]['y'], 'o', markersize=25, color=depot_color, zorder=5, label='Depot')
    customer_x = [node['x'] for i, node in nodes.items() if i != 0]
    customer_y = [node['y'] for i, node in nodes.items() if i != 0]
    ax.plot(customer_x, customer_y, 'o', markersize=25, color=customer_color, zorder=5, label='Customer',
            linestyle='None')

    for i, node in nodes.items():
        ax.text(node['x'], node['y'], str(i), color='white', ha='center', va='center', fontweight='bold', zorder=6,
                fontsize=10)
        if i != 0:
            ax.text(node['x'], node['y'] + 22, f"P: {node['p']}", ha='center', va='bottom',
                    fontsize=9, color='#312e81', bbox=dict(boxstyle="round,pad=0.3", fc="#eef2ff", ec="#c7d2fe", lw=1))

    # Helper to draw a single route
    def draw_single_route(route, color, style, width, label):
        if not route: return
        for i in range(len(route) - 1):
            from_node, to_node = nodes[route[i]], nodes[route[i + 1]]
            ax.arrow(from_node['x'], from_node['y'], to_node['x'] - from_node['x'], to_node['y'] - from_node['y'],
                     color=color, length_includes_head=True, head_width=12, head_length=15, lw=width, zorder=2,
                     linestyle=style)
        # Add a line for the legend
        ax.plot([], [], color=color, linestyle=style, lw=width, label=label)

    # Draw routes
    draw_single_route(initial_route, initial_route_color, '--', 2, 'Initial Route')
    draw_single_route(improved_route, improved_route_color, '-', 2.5, 'Improved Route')

    title = "Initial & Improved Routes" if improved_route else "Initial Route" if initial_route else "Problem Nodes"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.margins(0.15)
    ax.legend()
    st.pyplot(fig)


# --- STREAMLIT UI ---

st.title("🚚 Interactive SPTP Heuristic Solver")

# Initialize session state
if 'nodes' not in st.session_state:
    st.session_state.nodes = {
        0: {"name": "Depot", "p": 0, "x": 50, "y": 150},
        1: {"name": "Customer 1", "p": 25, "x": 150, "y": 50},
        2: {"name": "Customer 2", "p": 5, "x": 350, "y": 150},
        3: {"name": "Customer 3", "p": 40, "x": 150, "y": 250},
    }
    st.session_state.travel_times = [
        [0, 10, 30, 15], [10, 0, 25, 20], [30, 25, 0, 20], [15, 20, 20, 0]
    ]
    st.session_state.initial_route = None
    st.session_state.improved_route = None

# --- SIDEBAR FOR PROBLEM EDITING ---
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("⚙️ Problem Editor")

    # Customer Editor
    st.subheader("Customers")
    customer_ids = sorted([k for k in st.session_state.nodes.keys() if k != 0])

    ui_customer_data = {}
    for i in customer_ids:
        col1, col2 = st.columns([3, 1])
        ui_customer_data[i] = col1.number_input(f"C{i} P-Time", min_value=0, value=st.session_state.nodes[i]['p'],
                                                key=f"p_{i}")
        if col2.button("🗑️", key=f"del_{i}", help=f"Remove Customer {i}"):
            if len(st.session_state.nodes) > 2:
                new_nodes = {}
                old_ids = sorted(st.session_state.nodes.keys())
                id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids) if old_id != i}

                new_size = len(id_map)
                new_travel_times = np.zeros((new_size, new_size), dtype=int).tolist()

                for old_id, new_id in id_map.items():
                    new_nodes[new_id] = st.session_state.nodes[old_id]

                for old_from, new_from in id_map.items():
                    for old_to, new_to in id_map.items():
                        new_travel_times[new_from][new_to] = st.session_state.travel_times[old_from][old_to]

                st.session_state.nodes = new_nodes
                st.session_state.travel_times = new_travel_times
                st.session_state.initial_route = None
                st.session_state.improved_route = None
                st.rerun()

    if st.button("➕ Add Customer"):
        new_id = max(st.session_state.nodes.keys()) + 1 if st.session_state.nodes else 1
        st.session_state.nodes[new_id] = {"name": f"Customer {new_id}", "p": 10, "x": random.randint(50, 350),
                                          "y": random.randint(50, 250)}
        old_size = len(st.session_state.travel_times)
        new_matrix = np.zeros((old_size + 1, old_size + 1), dtype=int).tolist()
        for r in range(old_size):
            for c in range(old_size):
                new_matrix[r][c] = st.session_state.travel_times[r][c]
        for i in range(old_size):
            rand_time = random.randint(5, 30)
            new_matrix[i][old_size] = rand_time
            new_matrix[old_size][i] = rand_time
        st.session_state.travel_times = new_matrix
        st.rerun()

    # Travel Matrix Editor
    with st.expander("Travel Time Matrix", expanded=False):
        node_ids = sorted(st.session_state.nodes.keys())
        df = pd.DataFrame(st.session_state.travel_times, columns=node_ids, index=node_ids)
        edited_df = st.data_editor(df, key="matrix_editor")

    if st.button("Update Problem", use_container_width=True, type="primary"):
        for i in ui_customer_data:
            st.session_state.nodes[i]['p'] = ui_customer_data[i]
        st.session_state.travel_times = edited_df.values.tolist()
        st.session_state.initial_route = None
        st.session_state.improved_route = None
        st.toast("✅ Problem updated!")

    st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN PANEL ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Route Visualization")
    plot_routes(st.session_state.initial_route, st.session_state.improved_route, st.session_state.nodes)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Solver Controls & Results")

    if st.button("Run Heuristic Solver", use_container_width=True):
        with st.spinner("Building initial route..."):
            st.session_state.initial_route = construct_initial_route(st.session_state.nodes,
                                                                     st.session_state.travel_times)
        with st.spinner("Improving route..."):
            st.session_state.improved_route = improve_route(st.session_state.initial_route, st.session_state.nodes,
                                                            st.session_state.travel_times)
        st.rerun()

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    if not st.session_state.initial_route:
        st.info("Edit the problem in the sidebar, click 'Update Problem', then 'Run Heuristic Solver'.")

    if st.session_state.improved_route:
        st.success("Heuristic solver finished!")
        timeline_improved = get_route_timeline(st.session_state.improved_route, st.session_state.nodes,
                                               st.session_state.travel_times)
        timeline_initial = get_route_timeline(st.session_state.initial_route, st.session_state.nodes,
                                              st.session_state.travel_times)

        st.markdown(f"**Final Route:** `{' → '.join(map(str, st.session_state.improved_route))}`")
        st.metric("Total Trip Time", f"{timeline_improved['total_time']:.1f} min",
                  delta=f"{timeline_improved['total_time'] - timeline_initial['total_time']:.1f} min vs Initial")
        st.metric("Total Wait Time", f"{timeline_improved['total_wait_time']:.1f} min")

        with st.expander("Show Improved Route Timeline"):
            st.code("\n".join(timeline_improved['details']))

    if st.session_state.initial_route:
        with st.expander("Show Initial Route Timeline"):
            timeline = get_route_timeline(st.session_state.initial_route, st.session_state.nodes,
                                          st.session_state.travel_times)
            st.markdown(f"**Initial Route:** `{' → '.join(map(str, st.session_state.initial_route))}`")
            if not st.session_state.improved_route:
                st.metric("Total Trip Time", f"{timeline['total_time']:.1f} min")
                st.metric("Total Wait Time", f"{timeline['total_wait_time']:.1f} min")
            st.code("\n".join(timeline['details']))

    st.markdown("</div>", unsafe_allow_html=True)
