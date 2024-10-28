"""
Nombre: PIDController.py
Autor: Oscar Franco
Versión: 1 (2024-10-17)
Descripción: Clase de controlador PID con opciones de autotuning por metodología
            Relay-Feedback test.
"""
from typing import Tuple, List
from pandas import Series
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class PIDController:
    """
    Controlador PID (Proporcional, Integral, Derivativo).

    """

    def __init__(self, time_sample: float, Kc: float, Ki: float = 0.0, Kd: float = 0.0, CO_min: float = 0.0, CO_max: float = 100.0):
        """
        Inicializa una nueva instancia de la clase PIDController.

        Parámetros
        ----------
        time_sample : float
            Tiempo de muestreo de la información suministrada al controlador
        Kc : float
            Constante proporcional.
        Ki : float, opcional
            Constante integral (por defecto es 0.0).
        Kd : float, opcional
            Constante derivativa (por defecto es 0.0).
        CO_min : float, opcional
            Valor mínimo permisible de salida del controlador (por defecto es 0.0).
        CO_max : float, opcional
            Valor máximo permisible de salida del controlador (por defecto es 100.0).

        """
        self.Ts = time_sample

        self.set_controller_gains(Kc, Ki, Kd)

        self.Ku = None
        self.Pu = None
        self.Kp = None
        self.taup = None
        self.td = None

        self.CO_MIN = CO_min
        self.CO_MAX = CO_max
        self.CO_RANGE = CO_max - CO_min

        self.set_controller_status(False)
        self.auto_tuning_status = False
        self.set_auto_tuning_time(2*60.0) # tiempo en segundos
        self.auto_tuning_current_time = 0
        self.auto_tuning_CO0 = 0
        self.auto_tuning_t = []
        self.auto_tuning_y = []

        self.restart_controller()

    def __str__(self):
        return f"<{self.__class__.__name__}> Controlador PID (Kp: {self.Kc}, Ki: {self.Ki}, Kd: {self.Kd})"
    
    def restart_controller(self) -> None:
        """
        Reinicia el controlador.

        Restablece los valores de los errores discretizados Ek2, Ek1 y Ek a 0.
        """
        self.Ek2 = 0
        self.Ek1 = 0
        self.Ek = 0


    def set_controller_gains(self, Kc: float, Ki: float = 0.0, Kd: float = 0.0) -> None:
        '''
        Actualizar las ganancias del controlador

        Parámetros
        ----------
        Kc : float
            Constante proporcional.
        Ki : float, opcional
            Constante integral (por defecto es 0.0).
        Kd : float, opcional
            Constante derivativa (por defecto es 0.0).
        
        '''
        self.Kc = Kc
        self.Ki = Ki
        self.Kd = Kd

    def set_controller_status(self, automatic: bool = True) -> None:
        '''
        Actualizar el estado del controlador

        Parámetros
        ----------
        automatic: bool
            Estado del controlador, True el control está en automático y False el controlador está en manual (por defecto es True).
        
        '''
        self.automatic = automatic

        if self.automatic:
            self.restart_controller()

    def calculate_CO(self, y: float, ysp: float, current_CO: float) -> float:
        """
        Calcula la salida del controlador (CO) basada en la entrada actual, el setpoint y la salida actual.
        El resultado depende de si el controlador se encuentra en automático o manual o se está ejecutando un autotuning.

        Parámetros
        ----------
        y : float
            Valor de la variable de proceso actual.
        ysp : float
            Setpoint o valor deseado de la variable de proceso.
        current_CO : float
            Valor actual de la salida del controlador.

        Returns
        -------
        float
            Nueva salida del controlador ajustada.
        
        """
        if self.automatic:
            new_CO = self.automatic_pid(y, ysp, current_CO)
        elif self.auto_tuning_status:
            new_CO = self.auto_tuning(y, ysp, current_CO)
        else:
            new_CO = current_CO

        # Limitar la salida del controlador al rango permitido
        new_CO = max(self.CO_MIN, min(self.CO_MAX, new_CO))

        return new_CO
    
    def automatic_pid(self, y: float, ysp: float, current_CO: float) -> float:
        """
        Calcula la salida del controlador (CO) con algoritmo PID basada en la entrada actual, el setpoint y la salida actual.

        Parámetros
        ----------
        y : float
            Valor de la variable de proceso actual.
        ysp : float
            Setpoint o valor deseado de la variable de proceso.
        current_CO : float
            Valor actual de la salida del controlador.

        Returns
        -------
        float
            Nueva salida del controlador ajustada.
        
        """
        self.Ek2 = self.Ek1
        self.Ek1 = self.Ek
        self.Ek = ysp - y

        # Anti Wind-Up
        anti_wind_up = 0 if (current_CO >= self.CO_MAX or current_CO <= self.CO_MIN) else 1

        q0 = self.Kc + anti_wind_up * self.Ts * self.Ki / 2 + self.Kd / self.Ts
        q1 = self.Kc - anti_wind_up * self.Ts * self.Ki / 2 + 2 * self.Kd / self.Ts
        q2 = self.Kd / self.Ts
        delta_CO = q0 * self.Ek - q1 * self.Ek1 + q2 * self.Ek2
        new_CO = current_CO + delta_CO

        return new_CO
    
    def set_auto_tuning_time(self, auto_tuning_max_time: float) -> None:
        """
        Establece el tiempo de autoajuste del controlador.
        
        Parámetros
        ----------
        auto_tuning_max_time: float
            El tiempo en segundos necesario para el autoajuste.

        """
        self.auto_tuning_max_time = auto_tuning_max_time

    def start_auto_tuning(self) -> None:
        """
        Inicia el proceso de autoajuste del controlador.
        
        Detiene el controlador y establece el estado de autoajuste a verdadero.

        """
        self.auto_tuning_current_time = 0
        self.set_controller_status(False)
        self.auto_tuning_status = True

    def stop_auto_tuning(self) -> None:
        """
        Detiene el proceso de autoajuste del controlador.
        
        Establece el estado de autoajuste a falso y reactiva el controlador en automático.

        """
        self.auto_tuning_status = False
        self.set_controller_status(True)

    def set_auto_tuning_CO0(self, new_CO0:float) -> None:
        """
        Establece el valor inicial de la salida del controlador antes de iniciar el proceso de auto tuning.
        
        Parámetros
        ----------
        new_CO0: float
            Valor de la salida del controlador inicial del proceso.
        
        """
        self.auto_tuning_CO0 = new_CO0
    
    def auto_tuning(self, y: float, ysp: float, current_CO: float) -> float:
        """
        Calcula la salida del controlador para un paso del test de Relay Feedback.

        Parámetros
        ----------
        y : float
            Valor de la variable de proceso actual.
        ysp : float
            Setpoint o valor deseado de la variable de proceso.
        current_CO : float
            Valor actual de la salida del controlador.

        Returns
        -------
        float
            Nueva salida del controlador ajustada.
        
        """
        if 0 < self.auto_tuning_current_time <= self.auto_tuning_max_time:
            ep = 0.5 if ysp == 0 else abs(ysp) * 0.05
            em = -ep
            h = 0.05 * self.CO_RANGE
            co0 = self.auto_tuning_CO0
            cop = co0 + h
            com = co0 - h
            e = ysp - y

            if current_CO == co0:
                new_CO = cop
            elif e > ep and current_CO != cop:
                new_CO = cop
            elif e < em and current_CO != com:
                new_CO = com
            else:
                new_CO = current_CO

            self.auto_tuning_t.append(self.auto_tuning_current_time)
            self.auto_tuning_y.append(y)

        elif self.auto_tuning_max_time < self.auto_tuning_current_time <= self.auto_tuning_max_time + self.Ts:
            new_CO = self.auto_tuning_CO0
            self.stop_auto_tuning()
            self.characterize_rft()
            self.calculate_controller_gains(self.Ku, self.Pu)
            self.auto_tuning_t = []
            self.auto_tuning_y = []
        else:
            new_CO = current_CO
        
        self.auto_tuning_current_time = self.auto_tuning_current_time + self.Ts

        return new_CO
    
    def calculate_derivatives(self, result_graph: bool = False) -> np.ndarray:
        """
        Calcula las derivadas numéricas de la salida del sistema y grafica si es necesario.

        Returns
        -------
        numpy Array
            Derivadas de la salida del proceso.
        
        """    
        dydt = np.zeros(len(self.auto_tuning_y))
        dydt[1:-1] = (self.auto_tuning_y[2:] - self.auto_tuning_y[:-2]) / (self.t[2:] - self.t[:-2])

        if result_graph:
            fig, ax = plt.subplots()
            ax.plot(self.auto_tuning_t, self.auto_tuning_y, label="y")
            twax = ax.twinx()
            twax.plot(self.auto_tuning_t, dydt, color='r', label="dydt")
            plt.legend(loc="upper left")
            plt.show()

        return dydt

    def characterize_wave_with_derivatives(self, result_graph: bool = False) -> Tuple[float, float, int]:
        """
        Caracteriza la onda utilizando derivadas numéricas para determinar la amplitud y el periodo.

        Returns
        -------
        float
            Amplitud de la onda.
        float
            Periodo de oscilación de la onda
        int
            Indice con la ubicación de uno de los picos de la onda
        
        """    
        y0 = self.auto_tuning_y[0]
        self.auto_tuning_y = self.auto_tuning_y[self.auto_tuning_t>self.auto_tuning_max_time/3]
        self.auto_tuning_t = self.auto_tuning_t[self.auto_tuning_t>self.auto_tuning_max_time/3]
        ypeak = [0] * len(self.auto_tuning_y)
        dydt = self.calculate_derivatives()

        for i in range(1, len(self.auto_tuning_y)):
            try:
                if -0.05 <= dydt[i] <= 0.05 and dydt[i+1] <= -0.05:
                    ypeak[i+1] = 1
            except IndexError:
                pass

        if result_graph:
            fig, ax = plt.subplots()
            ax.plot(self.t, self.auto_tuning_y, label="y")
            twax = ax.twinx()
            twax.plot(self.t, ypeak, color='r', label="y_peak")
            plt.legend(loc="upper left")
            plt.show()

        tPui = 0
        tPui1 = 0
        PuEncontrados = []
        Pumin = 0
        ubicacion_pico = 0

        for i in range(1, len(self.auto_tuning_y)):
            if ypeak[i] == 1:
                tPui1 = self.auto_tuning_t[i]
                PuEncontrados.append(tPui1 - tPui)
                Pumin = np.amin(PuEncontrados)
                if PuEncontrados[-1] == Pumin:
                    ubicacion_pico = i
                tPui = tPui1

        amplitud = self.auto_tuning_y[ubicacion_pico] - y0

        return amplitud, Pumin, ubicacion_pico
    
    def characterize_wave_with_fft(self, result_graph: bool = False)-> Tuple[float, float, int]:
        """
        Caracteriza la onda utilizando FFT para determinar el periodo y la amplitud.

        Returns
        -------
        float
            Amplitud de la onda.
        float
            Periodo de oscilación de la onda
        int
            Indice con la ubicación de uno de los picos de la onda
        
        """
        self.auto_tuning_t = np.array(self.auto_tuning_t)
        self.auto_tuning_y = np.array(self.auto_tuning_y)
        mask = self.auto_tuning_t > (self.auto_tuning_max_time/3)
        self.auto_tuning_y = self.auto_tuning_y[mask]
        self.auto_tuning_t = self.auto_tuning_t[mask]
        serie_yt = Series(data=self.auto_tuning_y, index=self.auto_tuning_t)

        # Calcular la frecuencia de la estacionalidad usando FFT
        fft_result = np.fft.fft(serie_yt)
        frequencies = np.fft.fftfreq(len(serie_yt), d=self.Ts)

        # Filtro de frecuencias positivas
        positive_frequencies = frequencies[frequencies > 0]
        positive_fft = fft_result[frequencies > 0]

        # Encontrar la frecuencia dominante ignorando el componente de frecuencia cero
        dominant_frequency_index = np.argmax(np.abs(positive_fft))
        dominant_frequency = positive_frequencies[dominant_frequency_index]

        period = 1 / dominant_frequency
        amplitude = (np.max(serie_yt) - np.min(serie_yt)) / 2

        if result_graph:
            print(f"Período de la estacionalidad: {period:.4f} segundos")
            print(f"Amplitud de la estacionalidad: {amplitude:.4f}")

        return amplitude, period, np.argsort(np.abs(positive_fft))[0]

    def characterize_wave_with_signal(self, filtrate_data: bool = False, result_graph: bool = False) -> Tuple[float, float, int]:
        """
        Caracteriza la onda utilizando la biblioteca scipy.signal. Tiene la posibilidad de usar filtros para suavizar los datos

        Returns
        -------
        float
            Amplitud de la onda.
        float
            Periodo de oscilación de la onda
        int
            Indice con la ubicación de uno de los picos de la onda
        
        """
        if filtrate_data:
            # Aplicar el filtro Savitzky-Golay para suavizar los datos
            window_length = 5
            polyorder = 2
            data_smooth = savgol_filter(self.auto_tuning_y, window_length, polyorder)
            plt.plot(self.auto_tuning_t, self.auto_tuning_y, label='Datos originales')
            plt.plot(self.auto_tuning_t, data_smooth, label='Datos suavizados', linestyle='--')
            plt.xlabel('t')
            plt.ylabel('y')
            plt.legend()
            plt.show()
            self.auto_tuning_y = data_smooth

        # Detección de picos y valles
        mask = self.auto_tuning_t > (self.auto_tuning_max_time/3)
        self.auto_tuning_y = self.auto_tuning_y[mask]
        self.auto_tuning_t = self.auto_tuning_t[mask]
        peaks, _ = find_peaks(self.auto_tuning_y)
        valleys, _ = find_peaks(-self.auto_tuning_y)

        # Asegurarse de que las longitudes de peaks y valleys coincidan
        min_len = min(len(peaks), len(valleys))

        # Calcular el periodo de la oscilación (diferencia entre picos consecutivos)
        if len(peaks) > 1:
            periods = np.diff(self.auto_tuning_t[peaks])
            mean_period = np.mean(periods)
        else:
            print('No se encontraron múltiples picos en la tendencia')
            mean_period = np.nan

        # Calcular la amplitud (diferencia entre pico y valle)
        amplitudes = (self.auto_tuning_y[peaks[:min_len]] - self.auto_tuning_y[valleys[:min_len]])/2
        mean_amplitude = np.mean(amplitudes)

        if result_graph:
            print(f"Periodo de oscilación: {mean_period:.2f} unidades de tiempo")
            print(f"Amplitud media de oscilación: {mean_amplitude:.2f} unidades de variable")

            plt.plot(self.auto_tuning_t, self.auto_tuning_y, label='Datos')
            plt.plot(self.auto_tuning_t[peaks], self.auto_tuning_y[peaks], "x", label='Picos')
            plt.plot(self.auto_tuning_t[valleys], self.auto_tuning_y[valleys], "o", label='Valles')
            plt.title('Detección de Picos y Valles')
            plt.xlabel('Tiempo')
            plt.ylabel('Variable')
            plt.legend()
            plt.show()

        return mean_amplitude, mean_period, peaks[-2]
    
    def characterize_rft(self, y: List[float] = None, t: List[float] = None) -> Tuple[float, float, float, float, float]:
        """
        Determina los parámetros utilizando diferentes métodos de caracterización.
        
        Returns
        -------
        float
            Ganancia última del lazo de control
        float
            Periodo última del lazo de control
        float
            Ganancia del proceso calculada usando la ganancia y periodo último del lazo de control
        float
            Tiempo caracteristico del proceso calculada usando la ganancia y periodo último del lazo de control
        float
            Tiempo muerto del proceso calculada usando la ganancia y periodo último del lazo de control

        """
        opcion_caracterizacion = input("Seleccione el método (1. FFT, 2. Signal, 3. Derivadas): ")

        self.auto_tuning_t = np.array(self.auto_tuning_t if y is None else t)
        self.auto_tuning_y = np.array(self.auto_tuning_y if t is None else y)

        if opcion_caracterizacion == '2':
            a, Pu, ubicacion_pico = self.characterize_wave_with_signal()
        elif opcion_caracterizacion == '3':
            a, Pu, ubicacion_pico = self.characterize_wave_with_signal()
        else:
            a, Pu, ubicacion_pico = self.characterize_wave_with_fft()

        #------------Tomado de "Getting More Information from Relay-Feedback Tests - Luyben"-------------------#

        h = 0.05 * self.CO_RANGE
        Ku = 4 * h / (a * np.pi)

        # Determinación del b que es el tiempo que toma en llegar a la mitad de la amplitud a tomando como punto de partida la ubicación de un pico (ubicacion_pico)
        half_amplitude_index = np.where((self.auto_tuning_y[ubicacion_pico::-1] - self.auto_tuning_y[0]) <= a/2)[0][0]
        b = self.auto_tuning_t[ubicacion_pico] - self.auto_tuning_t[ubicacion_pico - half_amplitude_index]

        F = 4 * b / Pu
        R = 10**(-5.2783 + 12.7147 * F - 9.8974 * F**2 + 2.6788 * F**3)

        omegau = 2 * np.pi / Pu
        func = lambda x: -omegau * R * x - np.arctan(omegau * x) + np.pi
        taup = float(fsolve(func, 0)[0])
        td = R * taup
        Kp = np.sqrt(1 + (omegau * taup)**2) / Ku

        #--------------------------------------------------------------------------------#

        self.Ku = Ku
        self.Pu = Pu
        self.Kp = Kp
        self.taup = taup
        self.td = td

        return Ku, Pu, Kp, taup, td
    
    def calculate_controller_gains(self, Ku, Pu, metodo='Tyreus - Luyben Ku PI') -> Tuple[float, float, float]:
        """
        Calcula las ganancias Kc, Ki, y Kd del controlador según el método, usando Ku y Pu.

        Returns
        -------
        float
            Ganancia proporcional del controlador calculada
        float
            Ganancia integral del controlador calculada
        float
            Ganancia derivativa del controlador calculada
        str
            Texto descriptivo del método usado para al cálculo de las ganancias del controlador
        
        """
        if metodo=='Zigler - Nicols Ku PI':
            # Zigler - Nicols PI
            Kc = Ku/2.2
            tauI = Pu/1.2
            tauD = 0
        elif metodo=='Zigler - Nicols Ku PID':
            # Zigler - Nicols PID
            Kc = Ku/1.7
            tauI = Pu/2
            tauD = Pu/8
        elif metodo=='Tyreus - Luyben Ku PI':
            # Tyreus - Luyben PI (td/tau<0.1)
            Kc = Ku/3.2
            tauI = Pu/0.45
            tauD = 0
        elif metodo=='Tyreus - Luyben Ku PID':
            # Tyreus - Luyben PID
            Kc = Ku/2.2
            tauI = Pu/0.45
            tauD = Pu/6.3

        Ki = Kc / tauI
        Kd = Kc * tauD

        self.set_controller_gains(Kc, Ki, Kd)

        return Kc, Ki, Kd