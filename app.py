import streamlit as st
from agent_engine import master_agent

st.set_page_config(layout="wide")
st.title("Newton SkillUP Autonomous Agent (v4.0 - DSPy ReAct)")

st.markdown(
    "Enter a complex, multi-hop strategic command. "
    "The agent will autonomously navigate the SQL, Vector, and Graph databases to compile a brief."
)

user_command = st.text_area("Strategic Command:", height=100)

if st.button("Deploy Autonomous Agent"):
    if user_command.strip():
        with st.status("Agent Deployed. Initiating Cognitive Loop...", expanded=True) as status_box:
            st.write("🧠 Analyzing intent and planning trajectory...")
            agent_response = master_agent(command=user_command)
            status_box.update(state="complete", expanded=False)

        st.markdown("### Executive Brief")
        st.info(agent_response.comprehensive_executive_brief)

        with st.expander("🔍 View Agent Trajectory & Audit Log"):
            trajectory = agent_response.trajectory
            step = 0
            while True:
                if f"thought_{step}" not in trajectory:
                    break
                st.markdown(f"**Step {step + 1}**")
                st.markdown(f"- **Thought:** {trajectory[f'thought_{step}']}")
                st.markdown(f"- **Tool:** `{trajectory[f'tool_name_{step}']}` with args `{trajectory[f'tool_args_{step}']}`")
                if f"observation_{step}" in trajectory:
                    st.markdown(f"- **Observation:** {trajectory[f'observation_{step}']}")
                st.divider()
                step += 1
    else:
        st.warning("Please enter a strategic command before deploying the agent.")
