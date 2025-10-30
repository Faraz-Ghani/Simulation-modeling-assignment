import streamlit as st
import pandas as pd
import numpy as np
from q7_autocorrelation import autocorrelation_test, parse_input_numbers

# Page setup
st.set_page_config(page_title="Q7: Autocorrelation Test", layout="wide")

st.title("Question 7: Autocorrelation Test")
st.markdown("### Test for Autocorrelation in Random Numbers")

# quick reminder of what the hypotheses mean
st.markdown("""
**Hypothesis:**
- H₀: ρᵢ,ₘ = 0 (numbers are independent)
- H₁: ρᵢ,ₘ ≠ 0 (numbers are dependent)
""")

# Sidebar inputs for parameters
st.sidebar.header("Test Parameters")

# basic inputs — starting position, lag, and significance level
i = st.sidebar.number_input("Starting position (i)", min_value=1, value=5, step=1)
m = st.sidebar.number_input("Lag (m)", min_value=1, value=5, step=1)
alpha = st.sidebar.number_input("Significance level (α)", min_value=0.01, max_value=0.99, value=0.05, step=0.01)

st.sidebar.info(f"Will test correlation between numbers separated by {m} lag positions.")
st.sidebar.markdown("**Note:** N = total numbers, M = N - m = pairs used in test")

# default dataset to test with (so user doesn’t have to type in numbers)
default_data = """0.63 0.28 0.30 0.42 0.97 0.05 0.71 0.63 0.17 0.86
0.61 0.19 0.94 0.64 0.84 0.54 0.56 0.57 0.09 0.99
0.01 0.10 0.69 0.38 0.93 0.85 0.68 0.14 0.18 0.84
0.19 0.71 0.44 0.72 0.95 0.28 0.96 0.51 0.50 0.89
0.66 0.31 0.50 0.33 0.89 0.54 0.73 0.76 0.62 0.92"""

# main section for input
st.header("Input Random Numbers")
col1, col2 = st.columns([2, 1])

with col1:
    # let user choose between default data and custom entry
    input_method = st.radio("Input Method:", ["Use Default Data", "Enter Custom Data"])
    
    if input_method == "Use Default Data":
        numbers_text = st.text_area("Random Numbers (space or comma separated):", 
                                    value=default_data, height=150)
    else:
        numbers_text = st.text_area("Random Numbers (space or comma separated):", 
                                    value="", height=150, 
                                    placeholder="Enter numbers separated by spaces or commas")

with col2:
    # quick summary of parameters for reference
    st.markdown("**Parameters:**")
    st.info(f"""
    - i = {i} (starting position)
    - m = {m} (lag)
    - α = {alpha} (significance)
    - M = N - m (computed automatically)
    """)

# Run the test
if st.button("Run Autocorrelation Test", type="primary"):
    if numbers_text.strip():
        numbers = parse_input_numbers(numbers_text)
        
        if len(numbers) == 0:
            st.error("No valid numbers found in input!")
        else:
            st.success(f"Parsed {len(numbers)} numbers from input (N = {len(numbers)})")
            
            # show numbers in a clean grid format for readability
            st.subheader("Input Data")
            num_cols = 10  # display 10 numbers per row
            num_rows = (len(numbers) + num_cols - 1) // num_cols
            padded_numbers = numbers + [None] * (num_rows * num_cols - len(numbers))
            reshaped = np.array(padded_numbers).reshape(num_rows, num_cols)
            
            grid_data = []
            for r in range(num_rows):
                row_data = {}
                for c in range(num_cols):
                    idx = r * num_cols + c
                    if idx < len(numbers):
                        row_data[f"Pos {idx+1}"] = f"{numbers[idx]:.2f}"
                    else:
                        row_data[f"Col {c+1}"] = ""
                grid_data.append(row_data)
            df_input = pd.DataFrame(grid_data)
            st.dataframe(df_input, use_container_width=True)

            # actually run the test now
            result = autocorrelation_test(numbers, i, m, alpha)

            if result['error']:
                st.error(result['error'])
            else:
                st.subheader("Test Results")
                st.success(f"**M = {result['M']}** (pairs formed from total N = {result['N']})")

                # key metrics up top for quick view
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sample Size (N)", result['N'])
                with col2:
                    st.metric("Lag (m)", result['m'])
                with col3:
                    st.metric("Number of Pairs (M)", result['M'])
                with col4:
                    st.metric("Mean (Ȳ)", f"{result['mean']:.4f}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Autocorrelation (rₘ)", f"{result['r_m']:.4f}")
                with col2:
                    st.metric("Z₀ Statistic", f"{result['Z0']:.4f}")
                with col3:
                    st.metric("Z Critical (±)", f"{result['Z_critical']:.4f}")
                with col4:
                    decision = "REJECT H₀" if result['reject_H0'] else "FAIL TO REJECT H₀"
                    st.metric("Decision", decision)

                # more detailed breakdown
                st.markdown("---")
                st.subheader("Detailed Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Test Statistics:**")
                    st.write(f"- Total numbers (N): {result['N']}")
                    st.write(f"- Starting position (i): {result['i']}")
                    st.write(f"- Lag (m): {result['m']}")
                    st.write(f"- Pairs used (M = N - m): {result['M']}")
                    st.write(f"- Mean (Ȳ): {result['mean']:.4f}")
                    st.write(f"- Autocorrelation coefficient (rₘ): {result['r_m']:.4f}")

                with col2:
                    st.markdown("**Decision:**")
                    st.write(f"- Significance level (α): {result['alpha']}")
                    st.write(f"- Z₀: {result['Z0']:.4f}")
                    st.write(f"- Z critical: ±{result['Z_critical']:.4f}")
                    st.write(f"- |Z₀| = {abs(result['Z0']):.4f}")
                    st.write(f"- |Z₀| > Z critical: {result['reject_H0']}")
                    
                    if result['reject_H0']:
                        st.error(f"**Reject H₀** → {result['conclusion']}")
                    else:
                        st.success(f"**Fail to Reject H₀** → {result['conclusion']}")

    else:
        st.warning("Please enter some numbers to test!")

# collapsible instructions at the bottom
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    1. **Set Parameters** from the sidebar:
       - **i**: Starting position in the list
       - **m**: Lag (how far apart the numbers are)
       - **α**: Significance level (default 0.05)
    
    2. **What the test does:**
       - Calculates the autocorrelation coefficient \( r_m \)
       - Then computes \( Z₀ = r_m \sqrt{N} \)
       - Compares |Z₀| with the critical Z value for your α
    
    3. **Decision:**
       - If |Z₀| > Z critical → Reject H₀ (autocorrelation exists)
       - If |Z₀| ≤ Z critical → Fail to reject H₀ (numbers are independent)
    """)
