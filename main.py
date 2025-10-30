"""
Команда 6 — Інтегрування: Оцінка загального виторгу та обсягу продажів магазину
Аналіз продажів кав'ярні з використанням чисельних методів інтегрування
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy.interpolate import interp1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------------------
# Налаштування стилю графіків — Леськів Максим
#---------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

#---------------------------------------------------------------
# Клас чисельного інтегрування — Шнайдер Олеся
#---------------------------------------------------------------
class NumericalIntegration:
    """Клас для чисельного інтегрування різними методами"""

    @staticmethod
    def left_rectangle(x, y):
        """Метод лівих прямокутників"""
        integral = 0
        for i in range(len(x) - 1):
            integral += y[i] * (x[i + 1] - x[i])
        return integral

    @staticmethod
    def right_rectangle(x, y):
        """Метод правих прямокутників"""
        integral = 0
        for i in range(1, len(x)):
            integral += y[i] * (x[i] - x[i - 1])
        return integral

    @staticmethod
    def midpoint_rectangle(x, y):
        """Метод середніх прямокутників"""
        integral = 0
        for i in range(len(x) - 1):
            mid_y = (y[i] + y[i + 1]) / 2
            integral += mid_y * (x[i + 1] - x[i])
        return integral

    @staticmethod
    def trapezoid(x, y):
        """Складений метод трапецій"""
        integral = 0
        for i in range(len(x) - 1):
            integral += (y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2
        return integral

    @staticmethod
    def simpson(x, y):
        """Складений метод Сімпсона"""
        if len(x) < 3:
            return NumericalIntegration.trapezoid(x, y)
        integral = 0
        n = len(x)
        for i in range(0, n - 2, 2):
            if i + 2 >= n:
                integral += NumericalIntegration.trapezoid(
                    [x[i], x[i + 1]], [y[i], y[i + 1]]
                )
                break
            h1 = x[i + 1] - x[i]
            h2 = x[i + 2] - x[i + 1]
            h = h1 + h2
            integral += (h / 6) * (y[i] + 4 * y[i + 1] + y[i + 2])
        return integral

#---------------------------------------------------------------
# Головний клас аналізу продажів — Стецик Олег
#---------------------------------------------------------------
class CoffeeSalesAnalyzer:
    """Головний клас для аналізу продажів"""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.hourly_data = None
        self.results = {}


 # Завантаження та підготовка даних — Юра Марчак
    def load_data(self):
        print("=" * 80)
        print("ЗАВАНТАЖЕННЯ ДАНИХ")
        print("=" * 80)
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Завантажено {len(self.df)} записів")
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['date'] = self.df['datetime'].dt.date
        print(f"✓ Період: {self.df['date'].min()} - {self.df['date'].max()}")
        print(f"✓ Унікальних днів: {self.df['date'].nunique()}")
        print(f"✓ Діапазон цін: {self.df['money'].min():.2f} - {self.df['money'].max():.2f} грн\n")


# Групування даних по годинах — Шаповалов Олександр
    def prepare_hourly_data(self):
        print("=" * 80)
        print("АГРЕГАЦІЯ ПОГОДИННИХ ДАНИХ")
        print("=" * 80)
        hourly = self.df.groupby('hour').agg({
            'money': ['sum', 'mean', 'count']
        }).reset_index()
        hourly.columns = ['hour', 'total_revenue', 'avg_price', 'sales_count']
        all_hours = pd.DataFrame({'hour': range(24)})
        self.hourly_data = all_hours.merge(hourly, on='hour', how='left').fillna(0)
        print(f"✓ Створено погодинну статистику")
        print(f"✓ Активних годин: {(self.hourly_data['sales_count'] > 0).sum()}\n")

    # Інтерполяція функцій S(t), p(t), R(t) — Припотнюк Влад
    def create_interpolation(self, dense_points=1000):
        active_data = self.hourly_data[self.hourly_data['sales_count'] > 0].copy()
        if len(active_data) < 2:
            return None, None, None
        x = active_data['hour'].values
        sales = active_data['sales_count'].values
        prices = active_data['avg_price'].values
        revenue = active_data['total_revenue'].values
        f_sales = interp1d(x, sales, kind='cubic', fill_value='extrapolate')
        f_prices = interp1d(x, prices, kind='cubic', fill_value='extrapolate')
        f_revenue = interp1d(x, revenue, kind='cubic', fill_value='extrapolate')
        x_dense = np.linspace(x.min(), x.max(), dense_points)
        return x_dense, {
            'sales': f_sales(x_dense),
            'prices': f_prices(x_dense),
            'revenue': f_revenue(x_dense)
        }, active_data

        # Обчислення інтегралів різними методами — Новодворський Роман
        def calculate_integrals(self):
            print("=" * 80)
            print("ОБЧИСЛЕННЯ ІНТЕГРАЛІВ")
            print("=" * 80)
            active_data = self.hourly_data[self.hourly_data['sales_count'] > 0].copy()
            x = active_data['hour'].values.astype(float)
            sales = active_data['sales_count'].values.astype(float)
            revenue = active_data['total_revenue'].values.astype(float)
            methods = {
                'Ліві прямокутники': NumericalIntegration.left_rectangle,
                'Праві прямокутники': NumericalIntegration.right_rectangle,
                'Середні прямокутники': NumericalIntegration.midpoint_rectangle,
                'Метод трапецій': NumericalIntegration.trapezoid,
                'Метод Сімпсона': NumericalIntegration.simpson,
            }
            results_list = []
            for name, func in methods.items():
                Q = func(x, sales)
                R = func(x, revenue)
                results_list.append({'Метод': name, 'Q (обсяг)': Q, 'R (виторг, грн)': R})
                print(f"{name:25s}: Q = {Q:8.4f}, R = {R:10.2f} грн")
            x_dense, funcs, _ = self.create_interpolation()
            if x_dense is not None:
                f_sales = interp1d(x, sales, kind='cubic', fill_value='extrapolate')
                f_revenue = interp1d(x, revenue, kind='cubic', fill_value='extrapolate')
                Q_quad, _ = integrate.quad(f_sales, x.min(), x.max())
                R_quad, _ = integrate.quad(f_revenue, x.min(), x.max())
                results_list.append({'Метод': 'SciPy quad (еталон)', 'Q (обсяг)': Q_quad, 'R (виторг, грн)': R_quad})
                print(f"{'SciPy quad (еталон)':25s}: Q = {Q_quad:8.4f}, R = {R_quad:10.2f} грн\n")
            self.results['integration'] = pd.DataFrame(results_list)

"""
Команда 6 — Інтегрування: Оцінка загального виторгу та обсягу продажів магазину
Аналіз продажів кав'ярні з використанням чисельних методів інтегрування
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy.interpolate import interp1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------------------
# Налаштування стилю графіків — Леськів Максим
#---------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

#---------------------------------------------------------------
# Клас чисельного інтегрування — Шнайдер Олеся
#---------------------------------------------------------------
class NumericalIntegration:
    """Клас для чисельного інтегрування різними методами"""

    @staticmethod
    def left_rectangle(x, y):
        """Метод лівих прямокутників"""
        integral = 0
        for i in range(len(x) - 1):
            integral += y[i] * (x[i + 1] - x[i])
        return integral

    @staticmethod
    def right_rectangle(x, y):
        """Метод правих прямокутників"""
        integral = 0
        for i in range(1, len(x)):
            integral += y[i] * (x[i] - x[i - 1])
        return integral

    @staticmethod
    def midpoint_rectangle(x, y):
        """Метод середніх прямокутників"""
        integral = 0
        for i in range(len(x) - 1):
            mid_y = (y[i] + y[i + 1]) / 2
            integral += mid_y * (x[i + 1] - x[i])
        return integral

    @staticmethod
    def trapezoid(x, y):
        """Складений метод трапецій"""
        integral = 0
        for i in range(len(x) - 1):
            integral += (y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2
        return integral

    @staticmethod
    def simpson(x, y):
        """Складений метод Сімпсона"""
        if len(x) < 3:
            return NumericalIntegration.trapezoid(x, y)
        integral = 0
        n = len(x)
        for i in range(0, n - 2, 2):
            if i + 2 >= n:
                integral += NumericalIntegration.trapezoid(
                    [x[i], x[i + 1]], [y[i], y[i + 1]]
                )
                break
            h1 = x[i + 1] - x[i]
            h2 = x[i + 2] - x[i + 1]
            h = h1 + h2
            integral += (h / 6) * (y[i] + 4 * y[i + 1] + y[i + 2])
        return integral

#---------------------------------------------------------------
# Головний клас аналізу продажів — Стецик Олег
#---------------------------------------------------------------
class CoffeeSalesAnalyzer:
    """Головний клас для аналізу продажів"""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.hourly_data = None
        self.results = {}


 # Завантаження та підготовка даних — Юра Марчак
    def load_data(self):
        print("=" * 80)
        print("ЗАВАНТАЖЕННЯ ДАНИХ")
        print("=" * 80)
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Завантажено {len(self.df)} записів")
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['date'] = self.df['datetime'].dt.date
        print(f"✓ Період: {self.df['date'].min()} - {self.df['date'].max()}")
        print(f"✓ Унікальних днів: {self.df['date'].nunique()}")
        print(f"✓ Діапазон цін: {self.df['money'].min():.2f} - {self.df['money'].max():.2f} грн\n")


# Групування даних по годинах — Шаповалов Олександр
    def prepare_hourly_data(self):
        print("=" * 80)
        print("АГРЕГАЦІЯ ПОГОДИННИХ ДАНИХ")
        print("=" * 80)
        hourly = self.df.groupby('hour').agg({
            'money': ['sum', 'mean', 'count']
        }).reset_index()
        hourly.columns = ['hour', 'total_revenue', 'avg_price', 'sales_count']
        all_hours = pd.DataFrame({'hour': range(24)})
        self.hourly_data = all_hours.merge(hourly, on='hour', how='left').fillna(0)
        print(f"✓ Створено погодинну статистику")
        print(f"✓ Активних годин: {(self.hourly_data['sales_count'] > 0).sum()}\n")

    # Інтерполяція функцій S(t), p(t), R(t) — Припотнюк Влад
    def create_interpolation(self, dense_points=1000):
        active_data = self.hourly_data[self.hourly_data['sales_count'] > 0].copy()
        if len(active_data) < 2:
            return None, None, None
        x = active_data['hour'].values
        sales = active_data['sales_count'].values
        prices = active_data['avg_price'].values
        revenue = active_data['total_revenue'].values
        f_sales = interp1d(x, sales, kind='cubic', fill_value='extrapolate')
        f_prices = interp1d(x, prices, kind='cubic', fill_value='extrapolate')
        f_revenue = interp1d(x, revenue, kind='cubic', fill_value='extrapolate')
        x_dense = np.linspace(x.min(), x.max(), dense_points)
        return x_dense, {
            'sales': f_sales(x_dense),
            'prices': f_prices(x_dense),
            'revenue': f_revenue(x_dense)
        }, active_data

        # Обчислення інтегралів різними методами — Новодворський Роман
        def calculate_integrals(self):
            print("=" * 80)
            print("ОБЧИСЛЕННЯ ІНТЕГРАЛІВ")
            print("=" * 80)
            active_data = self.hourly_data[self.hourly_data['sales_count'] > 0].copy()
            x = active_data['hour'].values.astype(float)
            sales = active_data['sales_count'].values.astype(float)
            revenue = active_data['total_revenue'].values.astype(float)
            methods = {
                'Ліві прямокутники': NumericalIntegration.left_rectangle,
                'Праві прямокутники': NumericalIntegration.right_rectangle,
                'Середні прямокутники': NumericalIntegration.midpoint_rectangle,
                'Метод трапецій': NumericalIntegration.trapezoid,
                'Метод Сімпсона': NumericalIntegration.simpson,
            }
            results_list = []
            for name, func in methods.items():
                Q = func(x, sales)
                R = func(x, revenue)
                results_list.append({'Метод': name, 'Q (обсяг)': Q, 'R (виторг, грн)': R})
                print(f"{name:25s}: Q = {Q:8.4f}, R = {R:10.2f} грн")
            x_dense, funcs, _ = self.create_interpolation()
            if x_dense is not None:
                f_sales = interp1d(x, sales, kind='cubic', fill_value='extrapolate')
                f_revenue = interp1d(x, revenue, kind='cubic', fill_value='extrapolate')
                Q_quad, _ = integrate.quad(f_sales, x.min(), x.max())
                R_quad, _ = integrate.quad(f_revenue, x.min(), x.max())
                results_list.append({'Метод': 'SciPy quad (еталон)', 'Q (обсяг)': Q_quad, 'R (виторг, грн)': R_quad})
                print(f"{'SciPy quad (еталон)':25s}: Q = {Q_quad:8.4f}, R = {R_quad:10.2f} грн\n")
            self.results['integration'] = pd.DataFrame(results_list)

    # Оцінка похибок методів — Катело Настя
    def calculate_errors(self):
        print("=" * 80)
        print("Аналіз похибок")
        print("=" * 80)
        df_results = self.results['integration']
        if 'SciPy quad (еталон)' in df_results['Метод'].values:
            ref_idx = df_results[df_results['Метод'] == 'SciPy quad (еталон)'].index[0]
        else:
            ref_idx = df_results[df_results['Метод'] == 'Метод трапецій'].index[0]
        Q_ref = df_results.loc[ref_idx, 'Q (обсяг)']
        R_ref = df_results.loc[ref_idx, 'R (виторг, грн)']
        df_results['Похибка Q (абс)'] = abs(df_results['Q (обсяг)'] - Q_ref)
        df_results['Похибка Q (%)'] = abs((df_results['Q (обсяг)'] - Q_ref) / Q_ref * 100)
        df_results['Похибка R (абс)'] = abs(df_results['R (виторг, грн)'] - R_ref)
        df_results['Похибка R (%)'] = abs((df_results['R (виторг, грн)'] - R_ref) / R_ref * 100)
        self.results['integration'] = df_results
        print("\nПорівняння з еталонним методом:")
        print(df_results.to_string(index=False))
        print()

