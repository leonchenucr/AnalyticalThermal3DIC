import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import numba as nb
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

start = time.time()

a = 0.024       # Length in X direction (m)
b = 0.024       # Length in Y direction (m)
n_l = 11        # Total number of layers
n_c = 6         # Heat source layer index
th = np.array([0.0008, 0.0009, 0.0012, 0.0013, 0.00131, 0.00206, 0.00221, 0.00521, 0.00531, 0.00731, 0.01031]) 

# Each row corresponds to one layer: [kx, ky, kz]
ka_aniso = np.array([
    [2, 2, 0.4],   
    [29.75, 29.75, 35.36],   
    [102, 102, 61.50],     
    [10, 10, 80.275],   
    [1.5, 1.5, 1.5],     
    [140, 140, 140],  
    [30, 30, 30],    
    [400, 400, 400],   
    [10, 10, 10],    
    [400, 400, 400],
    [400, 400, 400]
])

kx = ka_aniso[:, 0]  # Thermal conductivity in x direction
ky = ka_aniso[:, 1]  # Thermal conductivity in y direction
kz = ka_aniso[:, 2]  # Thermal conductivity in z direction

num_eigen = 30  
num_l = 400      

x = (np.linspace(0, b, num_l+1)[:-1] + np.linspace(0, b, num_l+1)[1:])/2
y = (np.linspace(0, a, num_l+1)[:-1] + np.linspace(0, a, num_l+1)[1:])/2
X, Y = np.meshgrid(x, y) 

data_g = np.loadtxt('comsol_s8_pd_1.txt', delimiter=',')*0.1   
data_f = np.loadtxt('combined_fd3.txt', delimiter=' ') 
data = np.loadtxt('sr2.7.txt')


cos_mx = np.zeros((num_eigen, num_l))  
cos_ny = np.zeros((num_eigen, num_l)) 

for m in range(num_eigen):
    coeff_x = (2.0 if m > 0 else 1.0) * (b / num_l) / a
    cos_mx[m, :] = np.cos(m * np.pi / a * x) * coeff_x

for n in range(num_eigen):
    coeff_y = (2.0 if n > 0 else 1.0) * (a / num_l) / b
    cos_ny[n, :] = np.cos(n * np.pi / b * y) * coeff_y

@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:], nb.float64[:,:]), 
         parallel=True, cache=True)
def compute_gmn(data_g, cos_mx, cos_ny):
    M = cos_mx.shape[0]
    N = cos_ny.shape[0]
    gmn = np.zeros((M, N))
    
    # Parallel outer loop
    for m in nb.prange(M):
        for n in nb.prange(N):
            total = 0.0
            # Traverse all grid points
            for i in range(data_g.shape[0]):    
                for j in range(data_g.shape[1]):
                    total += data_g[i, j] * cos_mx[m, j] * cos_ny[n, i]
            gmn[m, n] = total
    return gmn

gmn = compute_gmn(data_g, cos_mx, cos_ny)
fmn = np.zeros((num_eigen, num_eigen))

# Calculate Fourier coefficients for temperature boundary conditions
for m in range(num_eigen):
    # Integration in x direction (vectorized operation)
    fx = np.sum(data_f * cos_mx[m, :], axis=1) 
    
    # Integration in y direction (vectorized operation)
    for n in range(num_eigen):
        fmn[m, n] = np.sum(fx * cos_ny[n, :])
        
AB = np.zeros((num_eigen, num_eigen, n_l * 2))

# Pre-calculate anisotropic lambda values
m_indices = np.arange(num_eigen)
n_indices = np.arange(num_eigen)
M, N = np.meshgrid(m_indices, n_indices, indexing='ij')

# Anisotropic eigenvalue matrix, different for each layer
lamb_layers = np.zeros((num_eigen, num_eigen, n_l))
for layer in range(n_l):
    # Anisotropic eigenvalue calculation: λ² = (mπ/a)²(kx/kz) + (nπ/b)²(ky/kz)
    lamb_layers[:, :, layer] = np.sqrt(
        (M * np.pi / a)**2 * (kx[layer] / kz[layer]) + 
        (N * np.pi / b)**2 * (ky[layer] / kz[layer])
    )

# Handle m=0 and n=0 case
# Optimize sparse matrix construction
row_indices = []
col_indices = []
values = []
b_values = []
b_row_indices = []
b_col_indices = []

# Bottom surface adiabatic boundary condition: dT/dz = 0 at z=0
row_indices.append(0)
col_indices.append(0)
values.append(-1)

row_indices.append(0)
col_indices.append(1)
values.append(1)

b_row_indices.append(0)
b_col_indices.append(0)
b_values.append(0)

for i in range(1, n_l):
    # Equation 1: Heat flux continuity, using z-direction thermal conductivity
    row_indices.append(2*i-1)
    col_indices.append(2*i-1)
    values.append(kz[i-1])
    
    row_indices.append(2*i-1)
    col_indices.append(2*i+1)
    values.append(-kz[i])
    
    # Equation 2: Temperature continuity
    row_indices.append(2*i)
    col_indices.append(2*i-2)
    values.append(1)
    
    row_indices.append(2*i)
    col_indices.append(2*i-1)
    values.append(th[i-1])
    
    row_indices.append(2*i)
    col_indices.append(2*i)
    values.append(-1)
    
    row_indices.append(2*i)
    col_indices.append(2*i+1)
    values.append(-th[i-1])

# Top surface temperature boundary condition: T = fmn[0,0] at z=th[-1]
row_indices.append(2*n_l-1)
col_indices.append(2*n_l-2)
values.append(1)

row_indices.append(2*n_l-1)
col_indices.append(2*n_l-1)
values.append(th[-1])

# Right-hand side
b_row_indices.append(2*n_l-1)
b_col_indices.append(0)
b_values.append(fmn[0, 0])

# Heat source term
b_row_indices.append(2*(n_c-1)-1)
b_col_indices.append(0)
b_values.append(-gmn[0, 0]*th[n_c-2])

b_row_indices.append(2*(n_c-1))
b_col_indices.append(0)
b_values.append(-0.5*gmn[0, 0]/(kz[n_c-1])*th[n_c-2]**2)

b_row_indices.append(2*n_c-1)
b_col_indices.append(0)
b_values.append(gmn[0, 0]*th[n_c-1])

b_row_indices.append(2*n_c)
b_col_indices.append(0)
b_values.append(0.5*gmn[0, 0]/(kz[n_c-1])*th[n_c-1]**2)

# Construct sparse matrix
amatrix = csc_matrix((values, (row_indices, col_indices)), shape=(n_l*2, n_l*2))
bvector = csc_matrix((b_values, (b_row_indices, b_col_indices)), shape=(n_l*2, 1))

xvector = spsolve(amatrix, bvector)
AB[0, 0, :] = xvector

# Handle cases where m≠0 or n≠0
for m in range(num_eigen):
    for n in range(num_eigen):
        if m == 0 and n == 0:
            continue
        
        # Optimize sparse matrix construction
        row_indices = []
        col_indices = []
        values = []
        b_values = []
        b_row_indices = []
        b_col_indices = []
        
        # Bottom surface adiabatic boundary condition: dT/dz = 0 at z=0
        # For non-zero modes, this means -λA + λB = 0, i.e., A = B
        row_indices.append(0)
        col_indices.append(0)
        values.append(1)
        
        row_indices.append(0)
        col_indices.append(1)
        values.append(-1)
        
        b_row_indices.append(0)
        b_col_indices.append(0)
        b_values.append(0)
        
        # Internal nodes
        for i in range(1, n_l):
            # Use anisotropic eigenvalues for current layer
            current_lamb_prev = lamb_layers[m, n, i-1]
            current_lamb_curr = lamb_layers[m, n, i]
            
            exp_neg_prev = np.exp(-current_lamb_prev * th[i-1])
            exp_pos_prev = np.exp(current_lamb_prev * th[i-1])
            exp_neg_curr = np.exp(-current_lamb_curr * th[i-1])
            exp_pos_curr = np.exp(current_lamb_curr * th[i-1])
            
            # Equation 1: Heat flux continuity (z direction)
            row_indices.append(2*i-1)
            col_indices.append(2*i-2)
            values.append(-kz[i-1] * current_lamb_prev * exp_neg_prev)
            
            row_indices.append(2*i-1)
            col_indices.append(2*i-1)
            values.append(kz[i-1] * current_lamb_prev * exp_pos_prev)
            
            row_indices.append(2*i-1)
            col_indices.append(2*i)
            values.append(kz[i] * current_lamb_curr * exp_neg_curr)
            
            row_indices.append(2*i-1)
            col_indices.append(2*i+1)
            values.append(-kz[i] * current_lamb_curr * exp_pos_curr)
            
            # Equation 2: Temperature continuity
            row_indices.append(2*i)
            col_indices.append(2*i-2)
            values.append(exp_neg_prev)
            
            row_indices.append(2*i)
            col_indices.append(2*i-1)
            values.append(exp_pos_prev)
            
            row_indices.append(2*i)
            col_indices.append(2*i)
            values.append(-exp_neg_curr)
            
            row_indices.append(2*i)
            col_indices.append(2*i+1)
            values.append(-exp_pos_curr)
        
        # Top surface temperature boundary condition: T = fmn[m,n] at z=th[-1]
        current_lamb_last = lamb_layers[m, n, n_l-1]
        exp_neg_last = np.exp(-current_lamb_last * th[n_l-1])
        exp_pos_last = np.exp(current_lamb_last * th[n_l-1])
        
        row_indices.append(2*n_l-1)
        col_indices.append(2*n_l-2)
        values.append(exp_neg_last)
        
        row_indices.append(2*n_l-1)
        col_indices.append(2*n_l-1)
        values.append(exp_pos_last)
        
        # Right-hand side
        b_row_indices.append(2*n_l-1)
        b_col_indices.append(0)
        b_values.append(fmn[m, n])
        
        # Heat source term
        source_lamb = lamb_layers[m, n, n_c-1]
        b_row_indices.append(2*(n_c-1))
        b_col_indices.append(0)
        b_values.append(gmn[m, n]/(kz[n_c-1] * source_lamb**2))
        
        b_row_indices.append(2*n_c)
        b_col_indices.append(0)
        b_values.append(-gmn[m, n]/(kz[n_c-1] * source_lamb**2))
        
        # Construct sparse matrix
        amatrix = csc_matrix((values, (row_indices, col_indices)), shape=(n_l*2, n_l*2))
        bvector = csc_matrix((b_values, (b_row_indices, b_col_indices)), shape=(n_l*2, 1))
        
        xvector = spsolve(amatrix, bvector)
        AB[m, n, :] = xvector

end = time.time()
print(f"Calculation time: {end-start:.2f} seconds")

print("Calculating error with COMSOL...")
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
temp_comsol = data[:, 3]


layer_ends = th  # th represents top surface coordinates of each layer
layer_starts = np.concatenate([[0], th[:-1]])

# Determine layer index for each point
layer_idx = np.searchsorted(layer_ends, z, side='left')
layer_idx = np.clip(layer_idx, 0, len(layer_ends)-1)

kz_layer = kz[layer_idx]  
temp = np.zeros_like(z)

# Handle zero-order mode (m=0,n=0)
A_0 = AB[0, 0, 2 * layer_idx]
B_0 = AB[0, 0, 2 * layer_idx + 1]
temp += A_0 + B_0 * z

# Zero-order heat source term for heat source layer (layer n_c)
mask_nc = (layer_idx == (n_c-1))
temp[mask_nc] -= 0.5 * gmn[0, 0] / kz_layer[mask_nc] * z[mask_nc]**2

# Handle higher-order modes
for m in range(num_eigen):
    for n in range(num_eigen):
        if m == 0 and n == 0:
            continue
        
        # Use anisotropic eigenvalues
        lamb_local = lamb_layers[m, n, layer_idx]
        A = AB[m, n, 2 * layer_idx]
        B = AB[m, n, 2 * layer_idx + 1]
        
        # Exponential terms
        exp_neg = np.exp(-lamb_local * z)
        exp_pos = np.exp(lamb_local * z)
        
        # Spatial cosine terms
        cos_mx = np.cos(m * np.pi/a * x)
        cos_ny = np.cos(n * np.pi/b * y)
        
        # Mode contribution
        temp_contribution = (A * exp_neg + B * exp_pos) * cos_mx * cos_ny
        
        # Higher-order heat source term for heat source layer
        source_lamb = lamb_layers[m, n, n_c-1]
        temp_contribution[mask_nc] += (gmn[m, n] / (kz_layer[mask_nc] * source_lamb**2)) * cos_mx[mask_nc] * cos_ny[mask_nc]
        
        temp += temp_contribution

# Calculate error
error = np.abs(temp - temp_comsol)
mean_error = np.mean(error)
max_error = np.max(error)
print(f"Mean absolute error: {mean_error}")
print(f"Maximum absolute error: {max_error}")

print("Visualizing...")
# Calculate actual physical size ratio
dx, dy, dz = x.max()-x.min(), y.max()-y.min(), z.max()-z.min()
aspect_ratio = [dx, dy, dz]

# Temperature field visualization
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.view_init(elev=30, azim=225)
sc1 = ax1.scatter(x, y, z, c=temp, 
                 cmap='jet', s=10, alpha=0.6,
                 vmin=min(temp_comsol.min(), temp.min()), 
                 vmax=max(temp_comsol.max(), temp.max()))

cbar1 = fig1.colorbar(sc1, ax=ax1, 
                     shrink=0.5,
                     aspect=20,
                     pad=0.08)
cbar1.set_label('Temperature (K)', fontsize=12, rotation=90, labelpad=15)
cbar1.ax.tick_params(labelsize=10)

# 3D coordinate labels setup
ax1.set_title('3D-SOV', fontsize=14)
ax1.set_xlabel('x (m)', fontsize=12)
ax1.set_ylabel('y (m)', fontsize=12)
ax1.zaxis.set_rotate_label(False) 
ax1.set_zlabel('z (m)', fontsize=12, rotation=90)
ax1.set_box_aspect(aspect_ratio)

plt.tight_layout()
plt.savefig('calculated_temperature_modified_bc_11layers6.png', dpi=300, bbox_inches='tight')
plt.show()

# COMSOL temperature field visualization
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.view_init(elev=30, azim=225)
sc2 = ax2.scatter(x, y, z, c=temp_comsol, 
                 cmap='jet', s=10, alpha=0.6)

cbar2 = fig2.colorbar(sc2, ax=ax2, 
                     shrink=0.5, aspect=20, pad=0.08)
cbar2.set_label('Temperature (K)', fontsize=12, rotation=90, labelpad=15)
cbar2.ax.tick_params(labelsize=10)

ax2.set_title('COMSOL', fontsize=14)
ax2.set_xlabel('x (m)', fontsize=12)
ax2.set_ylabel('y (m)', fontsize=12)
ax2.zaxis.set_rotate_label(False) 
ax2.set_zlabel('z (m)', fontsize=12, rotation=90)
ax2.set_box_aspect(aspect_ratio)

plt.tight_layout()
plt.savefig('comsol_temperature_optimized_11layers6.png', dpi=300, bbox_inches='tight')
plt.show()

# Error distribution visualization
fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.view_init(elev=30, azim=225)
sc3 = ax3.scatter(x, y, z, c=error, 
                 cmap='jet', s=10, alpha=0.6,
                 vmin=0, vmax=max_error)

cbar3 = fig3.colorbar(sc3, ax=ax3,
                     shrink=0.5,
                     aspect=20,
                     pad=0.08)
cbar3.set_label('Absolute Error (K)', fontsize=12, rotation=90, labelpad=15)
cbar3.ax.tick_params(labelsize=10)

ax3.set_title('Error Distribution', fontsize=14)
ax3.set_xlabel('x (m)', fontsize=12)
ax3.set_ylabel('y (m)', fontsize=12)
ax3.zaxis.set_rotate_label(False) 
ax3.set_zlabel('z (m)', fontsize=12, rotation=90)
ax3.set_box_aspect(aspect_ratio)

plt.tight_layout()
plt.savefig('error_distribution_modified_bc_11layers6.png', dpi=300, bbox_inches='tight')

plt.show()
