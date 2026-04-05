import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
from matplotlib.patches import Ellipse

# --------------------------------------------------------
# NUMERICAL METHODS & MATRIX DECOMPOSITIONS
# --------------------------------------------------------
def custom_cholesky(A):
    """Computes the Cholesky Decomposition A = GG^T"""
    n = A.shape[0]
    G = np.zeros_like(A, dtype=float)
    for k in range(n):
        sum_sq = sum(G[k, p]**2 for p in range(k))
        val = A[k, k] - sum_sq
        if val <= 0:
            raise ValueError("Matrix is not positive definite!")
        G[k, k] = np.sqrt(val)
        for i in range(k+1, n):
            sum_prod = sum(G[i, p] * G[k, p] for p in range(k))
            G[i, k] = (A[i, k] - sum_prod) / G[k, k]
    return G

def classical_gram_schmidt(A):
    """Computes the QR decomposition using Classical Gram-Schmidt"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for k in range(n):
        z = A[:, k].astype(float)
        for i in range(k):
            R[i, k] = np.dot(Q[:, i], A[:, k])
            z = z - R[i, k] * Q[:, i]
        R[k, k] = la.norm(z)
        if R[k, k] > 1e-10:
            Q[:, k] = z / R[k, k]
    return Q, R

def solve_ls_normal_equations(A, b):
    """Solve Least Squares using Normal Equations & Cholesky"""
    C = A.T @ A
    d = A.T @ b
    G = la.cholesky(C, lower=True)
    y = la.solve_triangular(G, d, lower=True)
    x = la.solve_triangular(G.T, y, lower=False)
    return x

def solve_ls_svd(A, b):
    """Solve Least Squares using SVD Pseudo-inverse"""
    U, S, VT = la.svd(A, full_matrices=False)
    S_inv = np.diag(1 / S)
    A_pinv = VT.T @ S_inv @ U.T
    return A_pinv @ b

def solve_ls_qr(A, b):
    """Solve Least Squares using QR Decomposition"""
    Q, R = classical_gram_schmidt(A)
    rhs = Q.T @ b
    x = la.solve_triangular(R, rhs)
    return x

def parse_matrix(matrix_str):
    """Parses a comma or space-separated string into a NumPy array."""
    try:
        lines = matrix_str.strip().split('\n')
        matrix = [list(map(float, line.replace(',', ' ').split())) for line in lines if line.strip()]
        return np.array(matrix)
    except Exception:
        return None

def matrix_to_latex(A):
    """Generates LaTeX code for displaying a dynamic NumPy array."""
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.shape[0] > 10 or A.shape[1] > 10:
        return r"\text{Matrix too large to display}"
    latex_str = r"\begin{pmatrix} "
    for i in range(A.shape[0]):
        latex_str += " & ".join([f"{val:g}" for val in A[i]])
        if i < A.shape[0] - 1:
            latex_str += r" \\ "
    latex_str += r" \end{pmatrix}"
    return latex_str

# --------------------------------------------------------
# HELPER VISUALIZATIONS
# --------------------------------------------------------
def plot_svd_geometry(A):
    """Visualizes the Rotate-Scale-Rotate geometry of 2D SVD"""
    U, S, VT = la.svd(A)
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax in axs:
        ax.set_aspect('equal')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color='gray', linewidth=1)
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    t = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    
    v1, v2 = np.array([1, 0]), np.array([0, 1])
    
    # 1. Input Space
    axs[0].plot(circle[0], circle[1], 'b-', lw=2)
    axs[0].quiver(0, 0, VT[0,0], VT[0,1], angles='xy', scale_units='xy', scale=1, color='r', label=r'$v_1$')
    axs[0].quiver(0, 0, VT[1,0], VT[1,1], angles='xy', scale_units='xy', scale=1, color='g', label=r'$v_2$')
    axs[0].set_title(r"Input Space $\mathbf{x}$")
    axs[0].legend(loc='upper right')
    
    # 2. V^T Rotation
    rot_circle = VT @ circle
    axs[1].plot(rot_circle[0], rot_circle[1], 'b-', lw=2)
    axs[1].quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label=r'$\mathbf{V}^T v_1 = e_1$')
    axs[1].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label=r'$\mathbf{V}^T v_2 = e_2$')
    axs[1].set_title(r"Rotation $\mathbf{V}^T$")
    axs[1].legend(loc='upper right')

    # 3. Sigma Scaling
    Sigma = np.diag(S)
    scaled_circle = Sigma @ rot_circle
    axs[2].plot(scaled_circle[0], scaled_circle[1], 'b-', lw=2)
    axs[2].quiver(0, 0, S[0], 0, angles='xy', scale_units='xy', scale=1, color='r', label=r'$\sigma_1 e_1$')
    axs[2].quiver(0, 0, 0, S[1], angles='xy', scale_units='xy', scale=1, color='g', label=r'$\sigma_2 e_2$')
    axs[2].set_title(r"Scaling $\boldsymbol{\Sigma}$")
    axs[2].legend(loc='upper right')

    # 4. U Rotation (Final Output)
    out_circle = U @ scaled_circle
    axs[3].plot(out_circle[0], out_circle[1], 'b-', lw=2)
    axs[3].quiver(0, 0, U[0,0]*S[0], U[1,0]*S[0], angles='xy', scale_units='xy', scale=1, color='r', label=r'$\sigma_1 u_1$')
    axs[3].quiver(0, 0, U[0,1]*S[1], U[1,1]*S[1], angles='xy', scale_units='xy', scale=1, color='g', label=r'$\sigma_2 u_2$')
    axs[3].set_title(r"Rotation $\mathbf{U}$ (Output)")
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    return fig

def plot_orthogonal_projection(a_vec, u_vec):
    """Visualizes 2D orthogonal projection"""
    u_norm = u_vec / la.norm(u_vec)
    proj = np.dot(a_vec, u_norm) * u_norm
    residual = a_vec - proj
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(0, 0, a_vec[0], a_vec[1], angles='xy', scale_units='xy', scale=1, color='k', label='Vector a')
    ax.quiver(0, 0, u_norm[0]*5, u_norm[1]*5, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.3, label='Direction of q')
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='red', label='Proj of a onto q')
    ax.quiver(proj[0], proj[1], residual[0], residual[1], angles='xy', scale_units='xy', scale=1, color='green', label='Residual (z)')
    
    ax.set_xlim(-1, max(a_vec[0], u_norm[0]*5) + 1)
    ax.set_ylim(-1, max(a_vec[1], u_norm[1]*5) + 1)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Gram-Schmidt: Orthogonal Projection")
    return fig

# --------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------
st.set_page_config(page_title="Matrices & Least Squares", layout="wide")
st.title("Interactive Matrix Decompositions & Least Squares")
st.markdown("*By Dr Nouralden Mohammed*")

tab1, tab2, tab3 = st.tabs(["Matrix Decompositions", "Least Squares Solvers", "Curve Fitting Models"])

# --- TAB 1: MATRIX DECOMPOSITIONS ---
with tab1:
    st.header("Matrix Decompositions Explorer")
    decomp_type = st.radio("Select Decomposition to Explore:", ["Cholesky Decomposition", "Gram-Schmidt (QR)", "Singular Value Decomposition (SVD)"], horizontal=True)
    
    if decomp_type == "Cholesky Decomposition":
        st.markdown("""
        **Cholesky Decomposition** factors a Symmetric Positive Definite (SPD) matrix $\mathbf{A}$ into $\mathbf{A} = \mathbf{G}\mathbf{G}^T$, where $\mathbf{G}$ is lower triangular.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Input Symmetric Matrix A (rows separated by newlines):**")
            mat_str = st.text_area("Matrix A", "3, 6\n6, 37", key="chol_A", height=150)
            A_chol = parse_matrix(mat_str)
            
        with col2:
            if A_chol is not None:
                if A_chol.shape[0] != A_chol.shape[1]:
                    st.error("Cholesky Decomposition requires a square matrix.")
                else:
                    st.latex(r"\mathbf{A} = " + matrix_to_latex(A_chol))
                    try:
                        G = custom_cholesky(A_chol)
                        st.success("Matrix is Positive Definite!")
                        st.markdown("**Resulting Lower Triangular Matrix G:**")
                        st.latex(r"\mathbf{G} = " + matrix_to_latex(G))
                        
                        st.markdown("**Verification ($G G^T = A$):**")
                        A_recon = G @ G.T
                        st.latex(r"\mathbf{G}\mathbf{G}^T = " + matrix_to_latex(A_recon))
                    except ValueError as e:
                        st.error(f"Error: {e}. The matrix must be Symmetric Positive Definite.")
            else:
                st.warning("Please enter a valid numeric matrix.")
                
    elif decomp_type == "Gram-Schmidt (QR)":
        st.markdown("""
        **Classical Gram-Schmidt** constructs an orthonormal basis $\mathbf{Q}$ from the columns of $\mathbf{A}$, yielding the QR decomposition $\mathbf{A} = \mathbf{Q}\mathbf{R}$.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Input Matrix A (columns act as basis vectors):**")
            mat_str = st.text_area("Matrix A", "4, 2\n1, 4", key="qr_A", height=150)
            A_qr = parse_matrix(mat_str)
            
        with col2:
            if A_qr is not None:
                st.latex(r"\mathbf{A} = " + matrix_to_latex(A_qr))
                if A_qr.shape == (2, 2):
                    a_vec = A_qr[:, 1]
                    u_vec = A_qr[:, 0]
                    fig = plot_orthogonal_projection(a_vec, u_vec)
                    st.pyplot(fig, clear_figure=True)
                
                Q, R = classical_gram_schmidt(A_qr)
                st.markdown("**QR Decomposition Result:**")
                st.latex(r"\mathbf{Q} = " + matrix_to_latex(Q))
                st.latex(r"\mathbf{R} = " + matrix_to_latex(R))

    elif decomp_type == "Singular Value Decomposition (SVD)":
        st.markdown(r"""
        **SVD** factors any matrix $\mathbf{A}$ into $\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$. 
        Geometrically, it represents a Rotation ($\mathbf{V}^T$), a Scaling ($\boldsymbol{\Sigma}$), and another Rotation ($\mathbf{U}$).
        """)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Input Matrix A (m x n):**")
            mat_str = st.text_area("Matrix A", "1, 1\n0, 1.5", key="svd_A", height=150)
            A_svd = parse_matrix(mat_str)
            
        with col2:
            if A_svd is not None:
                st.latex(r"\mathbf{A} = " + matrix_to_latex(A_svd))
                U, S, VT = la.svd(A_svd)
                
                st.markdown("**Left Singular Vectors (U):**")
                st.latex(r"\mathbf{U} = " + matrix_to_latex(U))
                
                s_str = ", ".join([f"\\sigma_{i+1} = {S[i]:.4f}" for i in range(len(S))])
                st.markdown("**Singular Values:** $" + s_str + "$")
                
                st.markdown("**Right Singular Vectors (V^T):**")
                st.latex(r"\mathbf{V}^T = " + matrix_to_latex(VT))
                
                if A_svd.shape == (2, 2):
                    fig = plot_svd_geometry(A_svd)
                    st.pyplot(fig, clear_figure=True)

# --- TAB 2: LEAST SQUARES SOLVERS ---
with tab2:
    st.header("Comparing Least Squares Solution Methods")
    st.markdown("Solve the overdetermined system $\mathbf{A}\mathbf{x} = \mathbf{b}$ using three different numerical algorithms.")
    
    col3, col4 = st.columns([1, 2])
    with col3:
        st.markdown("**Design Matrix A (m x n):**")
        mat_str = st.text_area("Matrix A", "1, 2\n1, 4\n1, 6", key="ls_A", height=150)
        st.markdown("**Observation Vector b (m x 1):**")
        b_str = st.text_area("Vector b", "1\n1\n1", key="ls_b", height=100)
        
        A_ls_data = parse_matrix(mat_str)
        b_ls_data = parse_matrix(b_str)
        
    with col4:
        if A_ls_data is not None and b_ls_data is not None:
            b_ls_data = b_ls_data.flatten()
            if A_ls_data.shape[0] != len(b_ls_data):
                st.error("Error: Matrix A row count must match Vector b length.")
            else:
                st.markdown("### Results")
                
                # Method 1
                st.markdown("**1. Normal Equations + Cholesky Decomposition**")
                try:
                    x_ne = solve_ls_normal_equations(A_ls_data, b_ls_data)
                    st.latex(r"\mathbf{x}_{LS} = " + matrix_to_latex(x_ne))
                except Exception as e:
                    st.error(f"Normal Equations failed (requires full column rank for Cholesky): {e}")
                    
                # Method 2
                st.markdown("**2. Singular Value Decomposition (Pseudo-inverse)**")
                x_svd = solve_ls_svd(A_ls_data, b_ls_data)
                st.latex(r"\mathbf{x}_{LS} = " + matrix_to_latex(x_svd))
                
                # Method 3
                st.markdown("**3. QR Decomposition (Gram-Schmidt)**")
                x_qr = solve_ls_qr(A_ls_data, b_ls_data)
                st.latex(r"\mathbf{x}_{LS} = " + matrix_to_latex(x_qr))
                
                st.info("""
                **Observation:** While all three methods yield the same mathematical result, **SVD and QR are preferred** in practice. 
                The Normal Equations explicitly form $\mathbf{A}^T\mathbf{A}$, which squares the condition number of the matrix, 
                making it highly susceptible to numerical round-off errors for ill-conditioned problems!
                """)

# --- TAB 3: CURVE FITTING ---
with tab3:
    st.header("Least Squares Curve Fitting")
    st.markdown("Fit a continuous model $f(x)$ to noisy data points by minimizing the sum of squared residuals.")
    
    col5, col6 = st.columns([1, 2.5])
    with col5:
        st.markdown("**True Function (Hidden from Solver):**")
        st.latex("y = a_0 + a_1 x + a_2 x^2")
        true_a0 = st.slider("True a0", -5.0, 5.0, 3.0)
        true_a1 = st.slider("True a1", -5.0, 5.0, -2.0)
        true_a2 = st.slider("True a2", -2.0, 2.0, 0.5)
        
        noise_std = st.slider("Noise Level (Std Dev)", 0.0, 5.0, 1.0)
        n_points = st.slider("Number of Data Points", 5, 100, 30)
        
        fit_model = st.selectbox("Select Model to Fit:", ["Linear (deg 1)", "Quadratic (deg 2)", "Cubic (deg 3)", "Exponential (y = a e^{bx})", "Logarithmic (y = a + b ln(x))"])
        
    with col6:
        # Generate Data
        np.random.seed(42)
        x_data = np.linspace(0.1, 10, n_points)  # Start at 0.1 to avoid log(0)
        y_true = true_a0 + true_a1 * x_data + true_a2 * (x_data**2)
        y_noisy = y_true + np.random.randn(n_points) * noise_std
        
        # Fit Data
        if fit_model == "Exponential (y = a e^{bx})":
            # Linearize: ln(y) = ln(a) + bx
            # Note: Exponential fit only works if y_noisy > 0
            valid_idx = y_noisy > 0
            if sum(valid_idx) < 3:
                st.error("Too many negative values for an exponential fit. Reduce noise or change true parameters.")
                y_fit = np.zeros_like(x_data)
            else:
                A_fit = np.vstack([np.ones(sum(valid_idx)), x_data[valid_idx]]).T
                log_y = np.log(y_noisy[valid_idx])
                coeffs = la.solve(A_fit.T @ A_fit, A_fit.T @ log_y)
                a_est = np.exp(coeffs[0])
                b_est = coeffs[1]
                y_fit = a_est * np.exp(b_est * x_data)
                st.markdown(f"**Fitted Parameters:** $a = {a_est:.4f}, \quad b = {b_est:.4f}$")
        elif fit_model == "Logarithmic (y = a + b ln(x))":
            valid_idx = x_data > 0
            A_fit = np.vstack([np.ones(sum(valid_idx)), np.log(x_data[valid_idx])]).T
            coeffs = la.solve(A_fit.T @ A_fit, A_fit.T @ y_noisy[valid_idx])
            y_fit = np.zeros_like(x_data)
            y_fit[valid_idx] = coeffs[0] + coeffs[1] * np.log(x_data[valid_idx])
            st.markdown(f"**Fitted Parameters:** $a = {coeffs[0]:.4f}, \quad b = {coeffs[1]:.4f}$")
        else:
            deg = 1 if "Linear" in fit_model else (2 if "Quadratic" in fit_model else 3)
            # Build Design Matrix
            A_fit = np.vstack([x_data**i for i in range(deg + 1)]).T
            # Solve via Normal Equations (for simplicity here)
            coeffs = solve_ls_normal_equations(A_fit, y_noisy)
            y_fit = A_fit @ coeffs
            
            coeff_str = ", ".join([f"a_{i} = {c:.4f}" for i, c in enumerate(coeffs)])
            st.markdown(f"**Fitted Parameters:** ${coeff_str}$")
            
        # Plot
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(x_data, y_true, 'g--', lw=2, label="True Underlying Function", zorder=2)
        ax4.scatter(x_data, y_noisy, color='red', alpha=0.7, label="Noisy Data Observations", zorder=3)
        ax4.plot(x_data, y_fit, 'b-', lw=2, label=f"Least Squares Fit ({fit_model})", zorder=4)
        
        # Draw Residuals
        for i in range(n_points):
            ax4.plot([x_data[i], x_data[i]], [y_noisy[i], y_fit[i]], color='gray', linestyle=':', alpha=0.5)
            
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.set_title("Least Squares Curve Fitting")
        ax4.legend()
        ax4.grid(True, linestyle=":", alpha=0.6)
        
        st.pyplot(fig4, clear_figure=True)
        
        # Calculate and display R-squared Goodness of fit
        ss_res = np.sum((y_noisy - y_fit)**2)
        ss_tot = np.sum((y_noisy - np.mean(y_noisy))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        st.metric(label="Goodness of Fit ($R^2$)", value=f"{r2:.4f}")