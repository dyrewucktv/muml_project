import numpy as np
from sklearn.metrics import mutual_info_score


# difference based cmi estimation
def conditional_mutual_information(X, Y, Z):
    z_values = np.unique(Z)
    n_z_values = len(z_values)
    n = len(Z)

    cmi = 0

    for i in range(n_z_values):
        z_value_tmp = z_values[i]
        z_condition = (Z == z_value_tmp)

        X_z = X[z_condition]
        Y_z = Y[z_condition]

        mi_XY_z = mutual_info_score(X_z, Y_z)
        p_z = np.sum(z_condition) / n

        cmi += p_z * mi_XY_z

    return cmi
