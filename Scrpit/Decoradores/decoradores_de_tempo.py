# Usando decoradores: 
import time

def execution_time(function):
    
    def calculate_time():
        
        start_time = time.time()
        function()
        end_time = time.time()
        print(end_time - start_time)
        
    return calculate_time
    


@execution_time
def operation():
    d = 5+9+7
    print("Resultado da operação: ",d)
    
    
operation()