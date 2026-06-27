import numpy as np

# =============================================================================
# DEFINICION DE LA FUNCION PAGERANK UTILIZANDO EL METODO DE LAS POTENCIAS
# =============================================================================

# Funcion para calcular el vector PageRank utilizando el Metodo de las Potencias
def pageRank(M, d=0.85, tol=1e-8, maxIter=1000):
    M = np.asarray(M, dtype=float) # Asegura que M sea un array de tipo float
    n = M.shape[0] # Numero de nodos (paginas) en el grafo

    # Detectar que columnas de la matriz M suman exactamente cero (sumideros)
    isUnlinkedPages = (np.sum(M, axis=0) == 0)
    # Para cada columna sumidero, se reemplaza todas sus filas con el valor 1/n
    M[:, isUnlinkedPages] = 1.0 / n

    # Inicializar el vector de rangos v^(0) uniformemente
    # Al inicio, un usuario aleatorio tiene la misma probabilidad de estar en cualquier pagina.
    v = np.ones(n) / n
    
    # Bucle iterativo del Metodo de las Potencias
    for i in range(maxIter):
        vOld = v.copy()
        
        # Actualizacion del vector de rangos segun la ecuacion de PageRank:
        # v^(k+1) = d * M * v^(k) + ((1 - d) / n) * vector_de_unos
        v = d * np.dot(M, vOld) + ((1 - d) / n) * np.ones(n)

        # Re-normalizacion numerica para mantener suma(v)=1 ante error de redondeo.
        v = v / v.sum()
        
        # Calcular la norma L1 de la diferencia entre el vector actual y el anterior para medir la convergencia.
        error = np.linalg.norm(v - vOld, ord=1) 
        
        # Si la diferencia es menor a la tolerancia, el algoritmo ha convergido
        if error < tol:
                print(f"-> Convergencia alcanzada en la iteracion {i+1} con error {error:.2e}")
                return v


    print(f"-> No se alcanzo la convergencia despues de {maxIter} iteraciones")
    return v

# =============================================================================
# DEFINICION DE LOS CASOS DE PRUEBA (MATRICES M DE TRANSICION)
# =============================================================================

# grafo_1.
M1 = np.array([
    [0,   0, 0],
    [1/2, 0, 0],
    [1/2, 1, 0]
], dtype=float)

# grafo_2.
M2 = np.array([
    [0,   1/2, 0,   1/2],
    [1/2, 0,   1/2, 0  ],
    [0,   1/2, 0,   1/2],
    [1/2, 0,   1/2, 0  ]
], dtype=float)

# grafo_3.
M3 = np.array([
    [0,   1/2, 1/3, 0, 0,   0,   0,   0],
    [1/2, 0,   1/3, 0, 0,   0,   0,   0],
    [1/2, 1/2, 0,   1, 0,   0,   0,   0],
    [0,   0,   1/3, 0, 0,   0,   0,   0],
    [0,   0,   0,   0, 0,   1/2, 1/3, 0],
    [0,   0,   0,   0, 1/2, 0,   1/3, 0],
    [0,   0,   0,   0, 1/2, 1/2, 0,   1],
    [0,   0,   0,   0, 0,   0,   1/3, 0]
], dtype=float)

# grafo_4.
M4 = np.array([
    [0, 0, 1/2, 0, 0],
    [1, 0, 0,   0, 0],
    [0, 1, 0,   0, 0],
    [0, 0, 1/2, 0, 1],
    [0, 0, 0,   1, 0]
], dtype=float)

# grafo_5.
M5 = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 0]
], dtype=float)

# grafo_propuesto.
# Grafo Propuesto (6 nodos) con un ciclo (nodo 0) y nodo sin salida (nodo 1).
MPropuesto = np.array([
    [1/3, 0, 0,   1, 0, 0],  
    [1/3, 0, 1/3, 0, 0, 0],  
    [0,   0, 0,   0, 0, 1],  
    [1/3, 0, 1/3, 0, 0, 0],  
    [0,   0, 1/3, 0, 0, 0],  
    [0,   0, 0,   0, 1, 0]   
], dtype=float)

# =============================================================================
# EJECUCION Y PRESENTACION DE RESULTADOS
# =============================================================================
if __name__ == "__main__":
    casos = {
        "Grafo 1": M1,
        "Grafo 2": M2,
        "Grafo 3": M3,
        "Grafo 4": M4,
        "Grafo 5": M5,
        "Grafo Propuesto": MPropuesto
    }
    
    print("=================================================================")
    print("   EJECUCION DEL ALGORITMO PAGERANK - METODO DE LAS POTENCIAS    ")
    print("=================================================================\n")
    
    for nombre, matriz in casos.items():
        print(f"--- Resultados para: {nombre} ---")
        
        # Ejecucion del algoritmo con los parametros requeridos
        vectorRank = pageRank(matriz, d=0.85, tol=1e-8, maxIter=1000)
        
        # Imprimir los valores formateados de probabilidad
        print("Vector de rangos resultante:")
        for nodo, rank in enumerate(vectorRank):
            print(f"  Pagina {nodo}: {rank:.6f} ({rank*100:.2f}%)")
        print("-" * 65 + "\n")