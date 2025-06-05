import numpy as np
import pandas as pd
import os
import pickle
import warnings

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from hybparsimony import HYBparsimony, Population

from scores import *


# ----------------------- Borramos carpeta "mejores_modelos" al iniciar -----------------------
CARPETA_MODELOS = "mejores_modelos"
os.makedirs(CARPETA_MODELOS, exist_ok=True)
for fichero in os.listdir(CARPETA_MODELOS):
    ruta = os.path.join(CARPETA_MODELOS, fichero)
    try:
        if os.path.isfile(ruta):
            os.remove(ruta)
    except Exception:
        pass

# ----------------------- Preparación de datos y CSV -----------------------
warnings.filterwarnings("ignore")

FOLDER_DATASETS = './datasets/'
df = pd.read_csv("datasets.csv")
df = df[df['name_file'].apply(lambda fn: os.path.exists(FOLDER_DATASETS + fn))].reset_index(drop=True)


# --------------------- Parametros principales --------------
MAX_TOTAL_DEPTH = 4  # profundidad máxima permitida
NUM_RUNS = 30
MAX_ITERS = 200 
TAU_FINAL = 0.90
lista_datasets = df['name_file']



# Nombre de salida con timestamp
now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
OUTPUT_CSV = f'./resultados/resultados_binarios5_{now}.csv'
OUTPUT_MEDIA_CSV = f'./resultados/resultados_media_binarios5_{now}.csv'
LOCK_FILE = OUTPUT_CSV + '.lock'

# Cabecera del CSV
columns = [
    'dataset','nrows','ncols',
    'run','seed',
    'i_5CV_logloss','p_5CV_logloss',
    'i_TST_logloss','p_TST_logloss',
    'i_geo_complexity','p_geo_complexity',
    'i_nfs_complexity','p_nfs_complexity',
    'i_best_model','p_best_model',
    
    'i_d', 'p_d',
    'i_l', 'p_l',
    'i_f', 'p_f',
    'i_tree_nfs', 'p_tree_nfs',

    'i_tree_used_fs', 'p_tree_used_fs',
    'i_selected_fs', 'p_selected_fs'
]


# ----------------------- Funcion de un experimento -----------------------

def experimento(params):
    TAU = get_tau()
    run, name_file = params
    # Carga y partición
    X = pd.read_csv(FOLDER_DATASETS + name_file)
    nrows = X.shape[0]
    ncols = X.shape[1]

    y = X.pop('target_end')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1234 * run
    )
    total_feats = X.shape[1]
    input_names = X.columns

    # Modelo interpretable
    # -------------------->
    interpretable_cfg = {
        "estimator": DecisionTreeClassifier, 
        "complexity": interpretability_score,
        "criterion": {"value": "gini", "type": Population.CONSTANT},
        "splitter": {"value": "best", "type": Population.CONSTANT},
        "max_depth": {"range": (1, MAX_TOTAL_DEPTH), "type": Population.INTEGER},
        "min_samples_split": {"range": (2, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
        "min_samples_leaf": {"range": (1, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
        "max_features": {"value": None, "type": Population.CONSTANT},
        "random_state": {"value": 1234, "type": Population.CONSTANT},
        "ccp_alpha": {"range": (0, 1), "type": Population.FLOAT},
    }

    


    model_i = HYBparsimony(
        fitness=getFitness_custom(interpretable_cfg['estimator'], 
                           interpretable_cfg['complexity'], 
                           default_cv_score),
        algorithm=interpretable_cfg,
        features=input_names,
        #cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234),
        n_jobs=1,
        maxiter=MAX_ITERS,
        rerank_error=0.01,
        npart=50,
        keep_history=True,
        seed_ini=1234 * run,
        verbose=0
    )
    model_i.fit(X_train, y_train)

    # Predicciones y métricas (5-CV y test)
    try:
        preds_i = model_i.best_model.predict_proba(X_test[model_i.selected_features].values)[:, 1]
        i_5CV    = -round(model_i.best_score, 6)
        i_TST    = round(log_loss(y_test, preds_i), 6)
        nfs_i    = len(model_i.selected_features)
        i_geo    = interpretability_score(model_i.best_model, nfs_i, 
                                        X_train[model_i.selected_features].values,
                                        y_train, TAU)
        i_nfs_c  = decision_tree_complexity(model_i.best_model, nfs_i) // 1e9
    except:
        i_5CV = 99.9
        i_TST = 99.9
        nfs_i = 99.9
        i_geo = 99.9
        i_nfs_c = 99.9



    # Complejidades
    
    # Sacar datos del árbol para i_best_model
    tree_i   = model_i.best_model
    i_d      = tree_i.get_depth()
    i_l      = tree_i.get_n_leaves()
    i_f      = np.unique(tree_i.tree_.feature[tree_i.tree_.feature >= 0]).size
    i_best   = str(model_i.best_model)
    i_feats  = model_i.selected_features

    # Obtener los índices de las variables usadas en el árbol (ignorando los nodos que no son splits reales)
    used_feature_indices = tree_i.tree_.feature
    used_feature_indices = used_feature_indices[used_feature_indices >= 0]
    i_tree_used_feature_names = model_i.selected_features[np.unique(used_feature_indices)]


    # Guardar modelo interpretable
    dump_i = {
        "run": run, "dataset": name_file,
        "history": model_i.history,
        "best_model": model_i.best_model,
        "selected_features": i_feats,
        "tree_features": i_tree_used_feature_names
    }
    with open(f"mejores_modelos/i_{os.path.splitext(name_file)[0]}_run_{run:02d}.pkl", "wb") as f:
        pickle.dump(dump_i, f)

    # Modelo parsimonioso
    # -------------------
    parsimonious_cfg = {
        "estimator": DecisionTreeClassifier,
        "complexity": decision_tree_complexity,
        "criterion": {"value": "gini", "type": Population.CONSTANT},
        "splitter": {"value": "best", "type": Population.CONSTANT},
        "max_depth": {"range": (1, MAX_TOTAL_DEPTH), "type": Population.INTEGER},
        "min_samples_split": {"range": (2, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
        "min_samples_leaf": {"range": (1, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
        "max_features": {"value": None, "type": Population.CONSTANT},
        "random_state": {"value": 1234, "type": Population.CONSTANT},
        "ccp_alpha": {"range": (0, 1), "type": Population.FLOAT},
    }
    model_p = HYBparsimony(
        algorithm=parsimonious_cfg,
        features=input_names,
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234),
        n_jobs=1,
        maxiter=MAX_ITERS,
        rerank_error=0.01,
        npart=50,
        keep_history=True,
        seed_ini=1234 * run,
        verbose=0
    )
    model_p.fit(X_train, y_train)

    # Predicciones y métricas (5-CV y test)
    try:
        preds_p  = model_p.best_model.predict_proba(X_test[model_p.selected_features].values)[:, 1]
        p_5CV    = -round(model_p.best_score, 6)
        p_TST    = round(log_loss(y_test, preds_p), 6)
        # Complejidades
        nfs_p    = len(model_p.selected_features)
        p_geo    = interpretability_score(model_p.best_model, nfs_p, 
                                        X_train[model_p.selected_features].values,
                                        y_train, TAU)
        p_nfs_c  = decision_tree_complexity(model_p.best_model, nfs_p) // 1e9
    except:
        p_5CV = 99.9
        p_TST = 99.9
        nfs_p = 99.9
        p_geo = 99.9
        p_nfs_c = 99.9

    # Sacar datos del árbol para p_best_model
    tree_p   = model_p.best_model
    p_d      = tree_p.get_depth()
    p_l      = tree_p.get_n_leaves()
    p_f      = np.unique(tree_p.tree_.feature[tree_p.tree_.feature >= 0]).size

    p_best   = str(model_p.best_model)
    p_feats  = model_p.selected_features

    # Obtener los índices de las variables usadas en el árbol (ignorando los nodos que no son splits reales)
    used_feature_indices = tree_p.tree_.feature
    used_feature_indices = used_feature_indices[used_feature_indices >= 0]
    p_tree_used_feature_names = model_p.selected_features[np.unique(used_feature_indices)]

    # Guardar modelo parsimonioso
    dump_p = {
        "run": run, "dataset": name_file, 
        "history": model_p.history,
        "best_model": model_p.best_model,
        "selected_features": p_feats,
        "tree_features": p_tree_used_feature_names
    }
    with open(f"mejores_modelos/p_{os.path.splitext(name_file)[0]}_run_{run:02d}.pkl", "wb") as f:
        pickle.dump(dump_p, f)

    return {
        'dataset': name_file, 'run': run, 'seed': 1234 * run,
        'nrows':nrows, 'ncols':ncols,
        'i_5CV_logloss': i_5CV, 'p_5CV_logloss': p_5CV,
        'i_TST_logloss': i_TST, 'p_TST_logloss': p_TST,
        'i_geo_complexity': i_geo, 'p_geo_complexity': p_geo,
        'i_nfs_complexity': i_nfs_c, 'p_nfs_complexity': p_nfs_c,
        'i_best_model': i_best, 'p_best_model': p_best,
        
        'i_d': i_d, 'i_l': i_l, 'i_f': i_f,
        'p_d': p_d, 'p_l': p_l, 'p_f': p_f,

        'i_tree_nfs': len(i_tree_used_feature_names), 'p_tree_nfs': len(p_tree_used_feature_names),
        'i_tree_used_fs': i_tree_used_feature_names, 'p_tree_used_fs': p_tree_used_feature_names,
        'i_selected_fs': str(i_feats), 'p_selected_fs': str(p_feats)
    }

# ----------------------- Ejecucion en paralelo -----------------------
import sys
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from filelock import FileLock
from pandas import DataFrame


# df debe estar ya cargado arriba, p.e. con pandas.read_csv(...)
tasks = [(run, fn) for run in range(NUM_RUNS) for fn in lista_datasets]


def main():
    set_tau(TAU_FINAL)
    
     # Asume que 'experimento', 'tasks', 'OUTPUT_CSV', 'LOCK_FILE' y 'columns' están definidos en el módulo
    executor = ProcessPoolExecutor()

    # Handler para Ctrl+C: cierra el executor y sale
    def handler(signum, frame):
        print("\nRecibido SIGINT. Cerrando workers")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    resultados = []
    # Lanzar todas las tareas en paralelo
    futures = [executor.submit(experimento, t) for t in tasks]

    try:
        # tqdm sobre los futures completados
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Procesos completados",
            unit="tarea"
        ):
            res = fut.result()
            resultados.append(res)
            # Guarda resultados
            df_tmp = pd.DataFrame(resultados, columns=columns)
            df_tmp.to_csv(OUTPUT_CSV, index=False)
            
            # Calcular la media por dataset para las columnas restantes
            columnas_incluir = ['run', 'seed', 'nrows', 'ncols',
                                'i_5CV_logloss','p_5CV_logloss',
                                'i_TST_logloss','p_TST_logloss',
                                'i_geo_complexity','p_geo_complexity',
                                'i_nfs_complexity','p_nfs_complexity',
                                # 'i_best_model','p_best_model',
                                
                                'i_d', 'p_d',
                                'i_l', 'p_l',
                                'i_f', 'p_f',
                                'i_tree_nfs', 'p_tree_nfs',

                                # 'i_tree_used_fs', 'p_tree_used_fs',
                                # 'i_selected_fs', 'p_selected_fs'
                                ]
            try:
                df_tmp = df_tmp.query('i_5CV_logloss<99.0 & p_5CV_logloss<99.0').reset_index(drop=True)
                media_por_dataset = df_tmp.groupby('dataset')[columnas_incluir].mean().reset_index()
                media_redondeada = media_por_dataset.round(4)
                media_redondeada.to_csv(OUTPUT_MEDIA_CSV, index=False)
            except:
                print('No se ha podido hacer la media')


    except KeyboardInterrupt:
        print("\nInterrupción por teclado. Cerrando executor")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        # Asegurarse de liberar recursos
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
