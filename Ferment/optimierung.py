import optuna
import numpy as np
from scipy.integrate import solve_ivp
from differentialgleichung import ODE_Bioreactor_Monod
from numbers import Number
def objective(trial,t_start,t_end,y0,consts,t_span):
    
    
    try:
        #die zu optimierenden Parameter
        feed_S1 = trial.suggest_float('feed_S1', low=0, high=5, step=0.05)  

        # # Setze die initialen Werte in y0
        consts[19]  =   feed_S1
        
        sol = solve_ivp(ODE_Bioreactor_Monod, [t_start, t_end], y0, args=(consts,), t_eval=t_span)
        cx_max =  np.max(sol.y[0])  
        cp_min =  np.min(sol.y[3])  
        cs       =   y0[1]
        # Ziel: Maximiere cx und minimiere cp
        # Sicherstellen, dass cx > 20 g/L und cp < 10 g/L
        if cx_max <= 20:
            objective_value = float('inf')  # Ungültiger Wert, wenn cx <= 20
        elif cp_min >= 10:
            objective_value = float('inf')  # Ungültiger Wert, wenn cp >= 10
        else:
            # Ziel: Maximiere cx und minimiere cp
            objective_value = - cx_max + cp_min   # Negative Werte, da Optuna minimiert
            # Maximieren von cx und Minimieren von cp mit Gewichtung
            #objective_value = -cx_max + 0.1 * cp_min
            if sol.success:
                trial.set_user_attr("sol", sol)  # Speichert die Lösung im Trial
                trial.set_user_attr("cs",cs)
        
        return objective_value
    except Exception as e:
        print(f"Fehler in der Optimierung: {e}")
        return float('nan')  # oder einen anderen Platzhalter