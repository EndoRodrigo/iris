import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df):
    sns.pairplot(df, hue="species_name")
    plt.title("Pairplot de características del Iris")
    plt.show()

def check_missing_and_duplicates(df):
    print("Nulos:\n", df.isnull().sum())
    print("Duplicados:", df.duplicated().sum())
