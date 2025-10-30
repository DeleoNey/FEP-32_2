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
