import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import log_loss




def decision_tree_complexity(model, nFeatures, **kwargs):
    r"""
    Complexity function for Decision Tree models.

    Parameters
    ----------
    model : model
        The model from which the internal complexity is calculated.
    nFeatures : int
        The number of the selected features.
    **kwargs : 
        Other arguments.

    Returns
    -------
    int
        10^9·nFeatures + (number of leaves)

    """
    num_leaves = model.get_n_leaves()
    int_comp = np.min((1E09-1,num_leaves)) # More leaves more complex  
    return nFeatures*1E09 + int_comp



def interpretability_score(model, nFeatures, X_val, y_val, tau):
    """
    model     : DecisionTreeClassifier entrenado
    nFeatures : número de features activas (lo que te pasa HYBparsimony)
    
    Internamente, usaremos X_val, y_val y tau que ya están cerrados en un closure.
    """
    # # Obtengo los X_val e y_val que el closure capturó
    # X_val = interpretability_score._X_val
    # y_val = interpretability_score._y_val
    # tau   = interpretability_score._tau

    # 1) Indices de hoja para cada fila de X_val
    leaf_indices = model.apply(X_val)
    if len(leaf_indices)==0:
        return 99.9

    # 2) Armo un DataFrame para contar cobertura y pureza (CSS) de cada hoja
    df = pd.DataFrame({
        'leaf': leaf_indices,
        'y_true': y_val
    })
    counts = df.groupby('leaf').size().rename('count')

    # 3) Calculo CSS (log_loss de la constante mayoritaria) por cada hoja
    purezas = {}
    for leaf_id, group in df.groupby('leaf'):
        vals = group['y_true'].values
        p1 = vals.mean()
        p0 = 1 - p1
        # Etiqueta mayoritaria
        if p1 >= p0:
            p_const = np.array([1.0, 0.0])
        else:
            p_const = np.array([0.0, 1.0])
        
        pred_ohe = np.tile(p_const, (len(vals), 1))
        css = log_loss(vals, pred_ohe, labels=[0, 1])
        purezas[leaf_id] = css

    # 4) Ordeno las hojas por cobertura (count), descendente
    df_stats = pd.DataFrame({
        'count': counts,
        'css_leaf': pd.Series(purezas)
    }).sort_values('count', ascending=False)
    
    # 5) ¿Cuántas hojas necesito para cubrir >= tau * N? (sal si es cero)
    N = len(X_val)
    umbral = tau * N
    acumulado = 0
    K = 0
    soma_css = 0.0
    for idx, row in df_stats.iterrows():
        acumulado += row['count']
        K += 1
        soma_css += row['css_leaf']
        if acumulado >= umbral:
            break
    if K==0:
        return 99.9
    
    # 6) Pureza promedio de esas K hojas
    CSS_bar = soma_css / K
    eps = 1e-6
    if CSS_bar >= 1e6:
        CSS_bar = 1e6-eps  # Max val of CSS
    
    # 7) Métrica final: cuanto más pequeña, más interpretable
    #    por ejemplo, puedes usar simplemente K + CSS_bar
    #    El numero de reglas tiene más peso. Si tienen el mismo numero de reglas, CSS_bar cuanto más pequeño mejor.
    return (K*1e6) + CSS_bar




def interpretability_score_weighted(model, nFeatures, X_val, y_val, tau):
    """
    Versión con cobertura ponderada.
    - Cada muestra recibe el peso que usaste al entrenar (class_weight='balanced').
    - El umbral tau se aplica sobre la suma de pesos, no sobre el recuento de filas.
    """
    # ------------------------------------------------------------------
    # 0) Calculamos los pesos de clase EXACTAMENTE como los usa sklearn
    #     class_weight = 'balanced'  ?  n_total / (n_clases * n_clase)
    # ------------------------------------------------------------------
    classes = np.unique(y_val)
    w_vec   = compute_class_weight(class_weight="balanced",
                                   classes=classes,
                                   y=y_val)
    class_w = dict(zip(classes, w_vec))          # {0: w0, 1: w1}


    # 1) Índice de la hoja para cada fila de X_val
    leaf_indices = model.apply(X_val)
    if len(leaf_indices) == 0:
        return 9999.9*1e6

    # 2) DataFrame con etiqueta y peso de cada fila
    y_series = pd.Series(y_val)             # ? conversión clave
    df = pd.DataFrame({
        "leaf": leaf_indices,
        "y_true": y_val,
        "w": y_series.map(class_w)          # peso según su clase
    })

    # 3) Cobertura ponderada: suma de pesos por hoja
    weight_by_leaf = df.groupby("leaf")["w"].sum()

    # 4) Pureza (CSS) igual que antes  no cambia
    purezas = {}
    for leaf_id, g in df.groupby("leaf"):
        vals = g["y_true"].values
        p1   = vals.mean()
        p0   = 1 - p1
        p_const = np.array([1.0, 0.0]) if p1 >= p0 else np.array([0.0, 1.0])
        pred = np.tile(p_const, (len(vals), 1))
        purezas[leaf_id] = log_loss(vals, pred, labels=[0, 1])

    # 5) Ordeno hojas por cobertura ponderada, descendente
    df_stats = pd.DataFrame({
        "w_sum": weight_by_leaf,
        "css_leaf": pd.Series(purezas)
    }).sort_values("w_sum", ascending=False)

    # 6) ¿Cuántas hojas necesito para cubrir tau · (peso total)?
    total_w   = df["w"].sum()
    threshold = tau * total_w
    acc_w = 0.0
    K = 0
    css_sum = 0.0
    for _, row in df_stats.iterrows():
        acc_w  += row["w_sum"]
        K      += 1
        css_sum += row["css_leaf"]
        if acc_w >= threshold:
            break
    if K == 0:
        return 9999.9*1e6

    CSS_bar = css_sum / K
    return (K * 1e6) + CSS_bar







# Global Constant
TAU = None
def set_tau(valor):
    global TAU
    TAU = valor

def get_tau():
    global TAU
    return TAU

def default_cv_score(estimator, X, y):
            return cross_val_score(estimator, X, y, 
                                   cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234),
                                   scoring='neg_log_loss', n_jobs=1)

def getFitness_custom(algorithm, complexity, custom_eval_fun, ignore_warnings = True):
    if algorithm is None:
        raise Exception("An algorithm function must be provided!!!")
    if complexity is None or not callable(complexity):
        raise Exception("A complexity function must be provided!!!")


    def fitness(cromosoma, **kwargs):
        global TAU
        if "pandas" in str(type(kwargs["X"])):
            kwargs["X"] = kwargs["X"].values
        if "pandas" in str(type(kwargs["y"])):
            kwargs["y"] = kwargs["y"].values

        X_train = kwargs["X"]
        y_train = kwargs["y"]
            
        try:
            # Extract features from the original DB plus response (last column)
            data_train_model = X_train[: , cromosoma.columns]

            if ignore_warnings:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"

            # train the model
            aux = algorithm(**cromosoma.params)
            fitness_val = custom_eval_fun(aux, data_train_model, y_train).mean()
            modelo = algorithm(**cromosoma.params).fit(data_train_model, y_train)

            # Reset warnings to default values
            warnings.simplefilter("default")
            os.environ["PYTHONWARNINGS"] = "default"
            # El híbrido funciona de forma que cuanto más alto es mejor. Por tanto, con RMSE deberíamos trabajar con su negación.
            return np.array([fitness_val, complexity(modelo, np.sum(cromosoma.columns),
                                                     data_train_model, y_train,
                                                     tau=TAU)]), modelo
        except Exception as e:    
            print(e)
            return np.array([-np.inf, np.inf]), None
    return fitness



from sklearn.utils import compute_class_weight






# def geo_complexity(model, nFeatures, **kwargs):
#     """
#     Complejidad geométrica: media geométrica de
#       (profundidad/MaxDepth),
#       (#hojas/2^profundidad) y
#       (#features/total_features).
#     Evita división por cero con validaciones.
#     """
#     tree_depth  = model.get_depth()
#     tree_leaves = model.get_n_leaves()
#     # número de features usado internamente
#     tree_feats  = np.unique(model.tree_.feature[model.tree_.feature >= 0]).size
    
#     # Evitar divisor 0 para profundidad
#     d = tree_depth / MAX_TOTAL_DEPTH

#     # Evitar 2**0=1 está bien, pero por si depth<0
#     denom_l = 2 ** tree_depth if tree_depth >= 0 else 1
#     l = tree_leaves / denom_l
    
#     # total_features viene en kwargs, si no o cero, usar tree_feats o 1
#     total_feats = 20 # Maximo numero de features interpretables
#     f = tree_feats / total_feats
#     if f > 1.0:
#         f = 1.0

#     # producto y raíz cúbica, si negativo o cero devuelve 0
#     prod = d * l * f
#     # print(d, l, f)
#     if prod > 0.0:
#         prod =  prod ** (1/3) if prod > 0 else 0
#     else:
#         prod = 1.0
#     return prod










# def css_score(model, X, y, tau=0.90, alpha=1.0, beta=1.0):
#     """
#     Coverage Simplicity Score para árboles sklearn.
#     """
#     from sklearn.metrics import log_loss
#     import numpy as np
    
#     leaf_ids = model.apply(X)                 # hoja por instancia
#     unique_leafs = np.unique(leaf_ids)
#     N = len(X)
    
#     # Información por hoja
#     leaf_stats = []
#     for leaf in unique_leafs:
#         idx = np.where(leaf_ids == leaf)[0]
#         cov = len(idx) / N
#         # prob predicha por la hoja
#         proba = model.predict_proba(X.iloc[idx])[:,1].mean()
#         err = log_loss(y.iloc[idx], np.repeat(proba, len(idx)))
#         leaf_stats.append((leaf, cov, err))
    
#     # Ordenar por cobertura desc, error asc
#     leaf_stats.sort(key=lambda t: (-t[1], t[2]))
    
#     # Acumular hasta tau
#     cov_acc = 0.0
#     K = 0
#     E = 0.0
#     for _, cov, err in leaf_stats:
#         if cov_acc >= tau: break
#         cov_acc += cov
#         K += 1
#         E += cov * err
    
#     K_max = len(unique_leafs)
#     # peor log-loss = -?p log p con p=0.5 en binario ? ln2  0.693
#     E_max = tau * np.log(2)
    
#     css = 1.0 / (alpha * K/K_max + beta * E/E_max)
#     return css, K, E, cov_acc


# def leaves_to_cover_tau(model, X, tau=0.90):
#     """
#     Devuelve el nº mínimo de hojas que necesitamos
#     para cubrir al menos un porcentaje tau del conjunto X.
    
#     Parameters
#     ----------
#     model : DecisionTreeClassifier / Regressor entrenado.
#     X      : pandas.DataFrame o ndarray con los datos.
#     tau    : float, fracción de cobertura deseada (0-1).

#     Returns
#     -------
#     k_tau  : int  -> nº de hojas
#     cover  : float -> cobertura real conseguida
#     """
#     import numpy as np
    
#     # hoja asignada a cada instancia
#     leaf_id = model.apply(X)
#     N = len(leaf_id)

#     # frecuencia de cada hoja (cobertura)
#     unique, counts = np.unique(leaf_id, return_counts=True)
#     coverage = counts / N                       # porcentaje que cubre cada hoja

#     # ordenar hojas por cobertura descendente
#     sorted_cov = np.sort(coverage)[::-1]        # de mayor a menor

#     # acumular hasta alcanzar tau
#     cum_cov = np.cumsum(sorted_cov)
#     k_tau = int((cum_cov < tau).sum() + 1)      # hojas mínimas
#     cover = cum_cov[k_tau-1]                    # cobertura obtenida

#     return k_tau, cover

# ----------------- Ejemplo de uso -----------------
#
# from sklearn.tree import DecisionTreeClassifier
# # supón que ya entrenaste tu árbol:
# model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
# 
# I, K, CSS = interpretability_score(model, X_valid, y_valid, tau=0.9)
# print(f"I = {I:.4f}  (K={K}, CSS¯={CSS:.4f})")
#
